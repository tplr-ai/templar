# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Adapted from https://github.com/bloc97/DeMo and NousResearch


# Global imports

import math
from typing import Generic, Literal, TypeAlias, TypeVar, cast, overload

import torch
from einops import rearrange

# ---------- type aliases ---------- #
IdxT: TypeAlias = torch.Tensor  # int16 indices
ValT: TypeAlias = torch.Tensor  # (possibly quantised) values
ShapeT: TypeAlias = tuple[int, ...]  # original tensor shape
TotK: TypeAlias = int  # size of the last dim
Shape4D = tuple[int, int, int, int]  # y, x, h, w

# (shift, scale, offset, lookup table, original dtype)
QuantParamsT: TypeAlias = tuple[torch.Tensor, float, int, torch.Tensor, torch.dtype]

# Boolean flag that propagates the chosen quantisation mode
Q = TypeVar("Q", Literal[True], Literal[False])


class TransformDCT:
    @torch.no_grad()
    def __init__(self, model, target_chunk, norm="ortho"):
        self.target_chunk = target_chunk

        self.shape_dict = dict()
        self.f_dict = dict()
        self.b_dict = dict()

        # Get all variants of model tensor sizes
        # Generate all possible valid DCT sizes for model tensors
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for s in p.shape:
                # Get the closest smallest divisor to the targeted DCT size
                sc = _get_smaller_split(s, self.target_chunk)
                self.shape_dict[s] = sc

                # Pregenerate DCT basis matrices
                if sc not in self.f_dict:
                    I = torch.eye(sc)  # noqa: E741
                    self.f_dict[sc] = _dct(I, norm=norm).to(p.dtype).to(p.device)
                    self.b_dict[sc] = _idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, jb, ld -> ...ikbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ibjd", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y h x w", h=n1, w=n2)
            if use_dct:
                x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            if use_dct:
                x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor, *, use_dct: bool = False):
        if len(x.shape) > 2:  # 2D weights
            if use_dct:
                n1 = x.shape[2]
                n2 = x.shape[3]
                n1w = self.b_dict[n1].to(x.device)
                n2w = self.b_dict[n2].to(x.device)
                self.b_dict[n1] = n1w
                self.b_dict[n2] = n2w

                x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y h x w -> (y h) (x w)")

        else:  # 1D weights
            if use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(x.device)
                self.b_dict[n1] = n1w

                x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class CompressDCT(Generic[Q]):
    """DCT-style sparsifier/compressor with optional 8-bit quantisation."""

    use_quantization: Q
    n_bins: int
    range_in_sigmas: int

    # ------------------------------------------------------------------ #
    # Constructor – two overloads so each instance "remembers" its mode
    # ------------------------------------------------------------------ #
    @overload
    def __init__(
        self: "CompressDCT[Literal[True]]",
        *,
        use_quantization: Literal[True] = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @overload
    def __init__(
        self: "CompressDCT[Literal[False]]",
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
        self.use_quantization = cast(Q, use_quantization)
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = (
                quantization_range  # Quantization range in standard deviations
            )

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return int(topk)

    # ------------------------------------------------------------------ #
    # compress – returns a 5-tuple *or* a 4-tuple, depending on the mode
    # ------------------------------------------------------------------ #
    @overload
    def compress(
        self: "CompressDCT[Literal[True]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK, QuantParamsT]: ...
    @overload
    def compress(
        self: "CompressDCT[Literal[False]]",
        x: torch.Tensor,
        topk: int,
    ) -> tuple[IdxT, ValT, ShapeT, TotK]: ...

    @torch.no_grad()
    def compress(self, x: torch.Tensor, topk: int):  # type: ignore[override]
        xshape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Limit topk to max size
        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # Cast idx to int16 for saving or transmission
        idx = idx_int64.to(torch.int16)

        # Apply 8-bit quantization if enabled
        if self.use_quantization:
            val, quant_params = self._quantize_values(val)
            return idx, val, xshape, totalk, quant_params

        return idx, val, xshape, totalk

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
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Cast back to int64 before using scatter/gather
        idx_int64 = idx.to(torch.int64)
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
        idx: torch.Tensor | list[torch.Tensor],
        val: torch.Tensor | list[torch.Tensor],
        xshape: ShapeT,
        totalk: int,
        quantize_params: QuantParamsT | list[QuantParamsT] | None = None,
        *,
        block_norms: torch.Tensor | None = None,
        normalise: bool = False,
        clip_norm: bool = True,
    ) -> torch.Tensor:
        if not isinstance(idx, list):
            idx = [idx]
        if not isinstance(val, list):
            val = [val]

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
            median_norm = torch.median(norms)
            clip_norm_val = torch.clamp(median_norm, min=1, max=10)

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

        # Concatenate everything
        idx_concat = torch.cat([i.to(p.device) for i in idx], dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        # Use decompress without quantization (since we already dequantized)
        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    @torch.no_grad()
    def _quantize_values(self, val: torch.Tensor) -> tuple[torch.Tensor, QuantParamsT]:
        offset = self.n_bins // 2  # 128 for 8-bit
        shift = val.mean()
        centered = val - shift

        std = centered.norm() / math.sqrt(centered.numel() - 1)
        scale = self.range_in_sigmas * std / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered.dtype, device=val.device)

        centered_fp32 = centered.to(torch.float32)
        qval = (
            (centered_fp32 / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        device = qval.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, qval.flatten().long(), centered_fp32.flatten())
        counts.scatter_add_(
            0, qval.flatten().long(), torch.ones_like(centered_fp32.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        qparams: QuantParamsT = (shift, float(scale), offset, lookup, val.dtype)
        return qval, qparams

    @torch.no_grad()
    def _dequantize_values(
        self, val: torch.Tensor, qparams: QuantParamsT
    ) -> torch.Tensor:
        shift, _, _, lookup, orig_dtype = qparams
        lookup = lookup.to(val.device) if isinstance(lookup, torch.Tensor) else lookup
        deq = lookup[val.long()] + shift
        return deq.to(orig_dtype)


# Code modified and sourced from https://github.com/zh217/torch-dct
def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n):
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


def _get_divisors(n):
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


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n
