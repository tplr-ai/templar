# The MIT License (MIT)
# © 2025 tplr.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from https://github.com/bloc97/DeMo and NousResearch


# Global imports

import math
from typing import Generic, Literal, Sequence, TypeAlias, TypeVar, cast, overload

import numpy as np
import torch
from einops import rearrange
from torch.distributed.tensor import DTensor as DT

import tplr

from .bits import decode_batch_rows, encode_batch_rows

# ─────────── type aliases ────────────────────────────────────────────────
ShapeT: TypeAlias = tuple[int, ...]
Shape4D = tuple[int, int, int, int]
TotK: TypeAlias = int
IdxT: TypeAlias = torch.Tensor  # stored as uint8 byte-stream (new codec)
QuantParamsT: TypeAlias = tuple[torch.Tensor, float, int, torch.Tensor, torch.dtype]
ValT: TypeAlias = torch.Tensor

_DEFAULT_B_CHOICES: tuple[int, ...] = (64, 128)

# Boolean flag that propagates the chosen quantisation mode
Q = TypeVar("Q", Literal[True], Literal[False])


class ChunkingTransformer:
    """
    A transformer for chunking tensors to enable more efficient gradient processing.
    """

    @torch.no_grad()
    def __init__(self, model, target_chunk):
        """
        Initialise the ChunkingTransformer.

        Args:
            model: The model whose parameters will be processed.
            target_chunk (int): The target size for tensor chunks.
        """
        self.target_chunk = target_chunk

        self.shape_dict = dict()

        # Get all variants of model tensor sizes
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for s in p.shape:
                # Get the closest smallest divisor to the target chunk size
                sc = _get_smaller_split(s, self.target_chunk)
                self.shape_dict[s] = sc

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a tensor by chunking.

        Args:
            x (torch.Tensor): The input tensor to encode.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            x = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            x = rearrange(x, "(x w) -> x w", w=n1)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor by un-chunking.

        Args:
            x (torch.Tensor): The input tensor to decode.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> (y h) (x w)")
        else:  # 1D weights
            x = rearrange(x, "x w -> (x w)")

        return x


class TopKCompressor(Generic[Q]):
    """
    A gradient sparsifier/compressor that uses Top-K selection and optional quantization.

    This class can be used to compress gradients by selecting the top-k largest values
    and optionally quantizing them to 8-bit integers for further size reduction.
    It supports both 1D and 2D tensors.
    """

    use_quantization: Q
    n_bins: int
    range_in_sigmas: int

    # ------------------------------------------------------------------ #
    # Constructor – two overloads so each instance "remembers" its mode
    # ------------------------------------------------------------------ #
    @overload
    def __init__(
        self: "TopKCompressor[Literal[True]]",
        *,
        use_quantization: Literal[True] = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @overload
    def __init__(
        self: "TopKCompressor[Literal[False]]",
        *,
        use_quantization: Literal[False] = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @torch.no_grad()
    def __init__(
        self,
        *,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None:
        """
        Initialise the TopKCompressor.

        Args:
            use_quantization (bool): Whether to use 8-bit quantization.
            quantization_bins (int): The number of bins for quantization.
            quantization_range (int): The quantization range in standard deviations.
        """
        self.use_quantization = cast(Q, use_quantization)
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )

    def _clamp_topk(self, x, topk) -> int:
        """
        Clamp the top-k value to be within the valid range and ensure it's even.

        Args:
            x (torch.Tensor): The input tensor.
            topk (int): The desired top-k value.

        Returns:
            int: The clamped and even top-k value.
        """
        topk = min(topk, x.shape[-1])
        topk = max(topk, 2)
        # Keep even by default (matches broader system expectations).
        topk = topk - (topk % 2)
        return int(topk)

    # ------------------------------------------------------------------ #
    # compress – returns a 5‑tuple (quant) or 4‑tuple (no quant)
    # ------------------------------------------------------------------ #
    @overload
    def compress(
        self: "TopKCompressor[Literal[True]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK, QuantParamsT]: ...
    @overload
    def compress(
        self: "TopKCompressor[Literal[False]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK]: ...

    @torch.no_grad()
    def compress(self, x: torch.Tensor, topk: int):  # type: ignore[override]
        """
        Compress a tensor using top-k selection and optional quantization.

        Args:
            x (torch.Tensor): The input tensor to compress.
            topk (int): The number of top values to select.

        Returns:
            A tuple containing the compressed data. The format depends on whether
            quantization is used.
        """
        if isinstance(x, DT):  # check for dtensors
            x = x.to_local()
        xshape = x.shape

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        # Top‑K
        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # Flatten to [rows, k] for the codec
        idx2d = idx_int64.reshape(-1, topk).contiguous()
        # GPU‑accelerated encode → bytes
        payload, _meta = encode_batch_rows(
            idx2d, C=totalk, B_choices=_DEFAULT_B_CHOICES
        )

        idx_bytes = torch.tensor(
            np.frombuffer(payload, dtype=np.uint8).copy(),
            dtype=torch.uint8,
            device="cpu",
        )

        if self.use_quantization:
            val, qparams = self._quantize_values(val)
            return idx_bytes, val, xshape, totalk, qparams
        return idx_bytes, val, xshape, totalk

    @torch.no_grad()
    def decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
        xshape: ShapeT,
        totalk: int,
        quantize_params: QuantParamsT | None = None,
    ) -> torch.Tensor:
        """
        Decompress a tensor from its sparse representation.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor): The indices of the non-zero values.
            val (torch.Tensor): The non-zero values.
            xshape (ShapeT): The original shape of the tensor.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (QuantParamsT, optional): Quantization parameters. Defaults to None.

        Returns:
            torch.Tensor: The decompressed tensor.
        """
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Decode indices
        if idx.dtype == torch.uint8:
            payload_bytes = idx.detach().cpu().numpy().tobytes()
            rows_list, C, _N = decode_batch_rows(payload_bytes)
            if C != totalk:
                raise ValueError(f"Index payload C={C} but expected {totalk}")
            k = val.shape[-1]
            if any(len(r) != k for r in rows_list):
                raise ValueError("Row-wise topk size mismatch in index payload")
            idx_int64 = torch.tensor(
                rows_list, dtype=torch.int64, device=p.device
            ).view(*val.shape)
        elif idx.dtype in (torch.int64, torch.long):
            idx_int64 = idx.to(p.device)
        else:
            raise ValueError(f"Unsupported index tensor dtype: {idx.dtype}")

        if val.dtype != x.dtype:
            val = val.to(dtype=x.dtype)

        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            xshape4 = cast(Shape4D, xshape)
            h_dim = xshape4[2]
            x = rearrange(x, "y x (h w) -> y x h w", h=h_dim)

        return x

    @torch.no_grad()
    def batch_decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor | Sequence[torch.Tensor],
        val: torch.Tensor | Sequence[torch.Tensor],
        xshape: ShapeT,
        totalk: int,
        quantize_params: Sequence[QuantParamsT] | None = None,
        *,
        block_norms: torch.Tensor | None = None,
        normalise: bool = False,
        clip_norm: bool = True,
    ) -> torch.Tensor:
        """
        Decompress a batch of sparse tensors and combine them.

        Args:
            p (torch.Tensor): A tensor with the target shape and device.
            idx (torch.Tensor | Sequence[torch.Tensor]): A sequence of indices for each tensor in the batch.
            val (torch.Tensor | Sequence[torch.Tensor]): A sequence of values for each tensor in the batch.
            xshape (ShapeT): The original shape of the tensors.
            totalk (int): The total number of elements in the original tensor's last dim.
            quantize_params (Sequence[QuantParamsT], optional): A sequence of quantization parameters. Defaults to None.
            block_norms (torch.Tensor, optional): Pre-computed norms for each block. Defaults to None.
            normalise (bool): Whether to normalise the values. Defaults to False.
            clip_norm (bool): Whether to clip the norms of the values. Defaults to True.

        Returns:
            torch.Tensor: The combined, decompressed tensor.
        """
        if quantize_params is not None and not isinstance(quantize_params, list):
            quantize_params = [quantize_params] * len(val)  # type: ignore[list-item]

        processed_vals: list[torch.Tensor] = []
        dequant_vals = None
        norms = None
        clip_norm_val = None
        if self.use_quantization and quantize_params:
            dequant_vals = [
                self._dequantize_values(v, quantize_params[i])
                for i, v in enumerate(val)
            ]
        if clip_norm:
            # If caller already supplied per-block norms, trust them.
            if block_norms is not None:
                norms = block_norms.to(p.device)
            else:
                vals_for_norm = dequant_vals if dequant_vals is not None else val
                norms = torch.stack(
                    [torch.norm(sparse_vals, p=2) for sparse_vals in vals_for_norm]
                )
            clip_norm_val = torch.median(norms)

        vals = dequant_vals if dequant_vals is not None else val
        for i, v in enumerate(vals):
            v = v.to(p.device)

            if normalise:
                eps = 1e-8
                if len(v.shape) == 3:  # 2D weights
                    l2_norm = torch.norm(v, p=2, dim=2, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 2:  # 1D weights (biases)
                    l2_norm = torch.norm(v, p=2, dim=1, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 1:  # Single values
                    l2_norm = torch.norm(v, p=2)
                    if l2_norm > eps:
                        v = v / l2_norm
            elif clip_norm and norms is not None and clip_norm_val is not None:
                current_norm = norms[i]
                clip_factor = torch.clamp(clip_norm_val / (current_norm + 1e-8), max=1)
                v = v * clip_factor
            processed_vals.append(v)

        # Unpack and concatenate indices
        unpacked_indices = []
        val_list = val if isinstance(val, Sequence) else [val]
        idx_list = idx if isinstance(idx, Sequence) else [idx]

        for i, i_data in enumerate(idx_list):
            v_data = val_list[i]
            if i_data.dtype == torch.uint8:
                rows, C, _N = decode_batch_rows(i_data.detach().cpu().numpy().tobytes())
                if C != totalk:
                    raise ValueError(f"Index payload C={C} but expected {totalk}")
                if any(len(r) != v_data.shape[-1] for r in rows):
                    raise ValueError(
                        "Row-wise topk size mismatch in index payload (batch)"
                    )
                idx_unpacked = torch.tensor(
                    rows, dtype=torch.int64, device=p.device
                ).view(*v_data.shape)
            elif i_data.dtype in (torch.int64, torch.long):
                idx_unpacked = i_data.to(p.device)
            else:
                raise ValueError(f"Unsupported index dtype in batch: {i_data.dtype}")
            unpacked_indices.append(idx_unpacked)

        idx_concat = torch.cat(unpacked_indices, dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    # -------------------- quantisation helpers ---------------------------
    @torch.no_grad()
    def _quantize_values(self, val: torch.Tensor) -> tuple[torch.Tensor, QuantParamsT]:
        """
        Quantize tensor values to 8-bit integers.

        Args:
            val (torch.Tensor): The tensor values to quantize.

        Returns:
            A tuple containing the quantized values (uint8) and the quantization parameters.
        """
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()
        centered = val - shift

        std = centered.norm() / math.sqrt(centered.numel() - 1)
        scale = self.range_in_sigmas * std / self.n_bins
        if (
            isinstance(scale, torch.Tensor)
            and (scale == 0 or torch.isnan(scale) or torch.isinf(scale))
        ) or (
            not isinstance(scale, torch.Tensor)
            and (scale == 0 or not math.isfinite(float(scale)))
        ):
            scale = torch.tensor(1.0, dtype=centered.dtype, device=val.device)
        centered_fp32 = centered.to(torch.float32)
        qval = ((centered_fp32 / scale + offset).round().clamp(0, self.n_bins - 1)).to(
            torch.uint8
        )
        device = qval.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, qval.flatten().long(), centered_fp32.flatten())
        counts.scatter_add_(
            0, qval.flatten().long(), torch.ones_like(centered_fp32.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        qparams: QuantParamsT = (shift, float(scale), int(offset), lookup, val.dtype)
        return qval, qparams

    @torch.no_grad()
    def _dequantize_values(
        self, val: torch.Tensor, qparams: QuantParamsT
    ) -> torch.Tensor:
        """
        Dequantize tensor values from 8-bit integers back to their original dtype.

        Args:
            val (torch.Tensor): The quantized values (uint8).
            qparams (QuantParamsT): The quantization parameters.

        Returns:
            torch.Tensor: The dequantized values.
        """
        if val.dtype == torch.uint8:
            shift, _scale, _offset, lookup, orig_dtype = qparams
            lookup = lookup.to(val.device)
            deq = lookup[val.long()] + shift
            val = deq.to(orig_dtype)
        return val

    def maybe_dequantize_values(
        self,
        vals: list[torch.Tensor],
        qparams: list[QuantParamsT],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """
        Dequantize a list of values if they are quantized.

        Args:
            vals (list[torch.Tensor]): A list of tensors that may be quantized.
            qparams (list[QuantParamsT]): A list of quantization parameters.
            device (torch.device): The device to move the tensors to.

        Returns:
            list[torch.Tensor]: A list of dequantized tensors.
        """
        if not isinstance(vals, (list, tuple)):
            vals = [vals]

        needs_dequantized = all([v.dtype == torch.uint8 for v in vals])
        if qparams is None or not needs_dequantized:
            return vals

        if (
            isinstance(qparams, tuple)
            and len(qparams) == 5  # potentially single or already 5 elements
            and not all([len(q) == 5 for q in qparams])  # already correctly formatted
        ):
            qparams = [qparams]

        if not isinstance(qparams, list):
            qparams = [qparams]

        vals_f32: list[torch.Tensor] = []
        for i, v in enumerate(vals):
            v = v.to(device)
            if v.dtype == torch.uint8:  # still quantised → decode
                if qparams[i] is None:
                    tplr.logger.warning(f"Missing quant_params for vals[{i}]]; skip.")
                    break
                qp = qparams[i]
                v = self._dequantize_values(v, qp).to(device)
            vals_f32.append(v)

        if len(vals_f32) != len(vals):  # some decode failed
            raise IndexError(
                f"Mismatch in val lengths: dequant({len(vals_f32)}) vs original({len(vals)})"
            )

        return vals_f32


def _get_prime_divisors(n: int) -> list[int]:
    """
    Get the prime divisors of a number.

    Args:
        n (int): The number to factorize.

    Returns:
        list[int]: A list of prime divisors.
    """
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n: int) -> list[int]:
    """
    Get all divisors of a number.

    Args:
        n (int): The number to get divisors for.

    Returns:
        list[int]: A sorted list of divisors.
    """
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n: int, close_to: int) -> int:
    """
    Find the largest divisor of n that is less than or equal to close_to.

    Args:
        n (int): The number to find a divisor for.
        close_to (int): The target value to be close to.

    Returns:
        int: The largest divisor of n that is <= close_to.
    """
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n
