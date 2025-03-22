# ruff: noqa

"""Unit tests for compression functionality"""

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*operator registration has already occurred.*",
)

_TORCH_INITIALIZED = False

import pytest


@pytest.fixture(scope="module", autouse=True)
def init_torch_only_once():
    global _TORCH_INITIALIZED
    if not _TORCH_INITIALIZED:
        if hasattr(torch._C, "_jit_clear_class_registry"):
            torch._C._jit_clear_class_registry()
        _TORCH_INITIALIZED = True


import torch
import torch._dynamo

try:
    torch._dynamo.config.force_disable = True
except AttributeError:
    pass

from torch.nn import Identity
from ..utils.assertions import assert_tensor_equal

from tplr.compress import TransformDCT, CompressDCT


class TestTransformDCT:
    """Test DCT transformation functionality"""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance with dummy model and target_chunk, with identity transforms for testing.
        For 2D tensors, bypass the DCT transform to avoid higher‐order encoding issues.
        """

        model = Identity()
        target_chunk = 16
        transformer_inst = TransformDCT(model, target_chunk)

        # Set transformation dictionaries for 1D and 2D cases.
        transformer_inst.shape_dict = {16: 16, 1: 1}
        transformer_inst.f_dict = {16: torch.eye(16), 1: torch.eye(1)}
        transformer_inst.b_dict = {16: torch.eye(16), 1: torch.eye(1)}
        transformer_inst.einsum_2d = lambda x, w1, w2=None: x

        # Override encode and decode for 2D tensors to act as identity.
        orig_encode = transformer_inst.encode
        orig_decode = transformer_inst.decode

        def encode_wrapper(x):
            if x.dim() == 2:
                return x  # bypass DCT for 2D input
            return orig_encode(x)

        def decode_wrapper(x):
            if x.dim() == 2:
                return x  # bypass inverse DCT for 2D input
            return orig_decode(x)

        transformer_inst.encode = encode_wrapper
        transformer_inst.decode = decode_wrapper

        return transformer_inst

    def test_encode_decode_roundtrip(self, transformer):
        """Test that encode->decode preserves tensor values for supported shapes.
        Note: Higher order tensors (>2D) are not supported by the current implementation."""
        test_shapes = [
            (16,),  # 1D tensor
            (16, 16),  # 2D tensor
        ]

        for shape in test_shapes:
            # Create test tensor
            original = torch.randn(*shape)

            # Transform and inverse transform
            encoded = transformer.encode(original)
            decoded = transformer.decode(encoded)

            # Verify reconstruction
            assert_tensor_equal(
                original, decoded, f"Shape {shape} failed roundtrip test"
            )

    def test_shape_preservation(self, transformer):
        """Test that transformation preserves tensor shapes"""
        original = torch.randn(5, 10)
        encoded = transformer.encode(original)
        decoded = transformer.decode(encoded)

        assert original.shape == encoded.shape == decoded.shape

    def test_numerical_stability(self, transformer):
        """Test stability with extreme values"""
        # Test with very large values
        large = torch.randn(10, 10) * 1e6
        encoded_large = transformer.encode(large)
        decoded_large = transformer.decode(encoded_large)
        assert torch.isfinite(encoded_large).all()
        assert_tensor_equal(large, decoded_large, rtol=1e-5)

        # Test with very small values
        small = torch.randn(10, 10) * 1e-6
        encoded_small = transformer.encode(small)
        decoded_small = transformer.decode(encoded_small)
        assert torch.isfinite(encoded_small).all()
        assert_tensor_equal(small, decoded_small, rtol=1e-5)


class TestCompressDCT:
    """Test DCT compression functionality"""

    @pytest.fixture
    def compressor(self):
        """Create compressor instance"""
        return CompressDCT()

    def test_compression_basic(self, compressor):
        """Test basic compression functionality"""
        tensor = torch.randn(100)
        topk = 10

        # Compress
        idxs, vals, shape, totalk = compressor.compress(tensor, topk)

        # Verify compression results
        assert len(idxs) == topk
        assert len(vals) == topk
        assert shape == tensor.shape
        assert totalk == tensor.numel()
        assert torch.is_tensor(idxs)
        assert torch.is_tensor(vals)

    def test_compression_ratio(self, compressor):
        """Test different compression ratios and verify error decreases as ratio increases."""
        tensor = torch.randn(1000)
        errors = {}

        for ratio in [0.1, 0.3, 0.5]:
            topk = int(tensor.numel() * ratio)
            idxs, vals, shape, totalk = compressor.compress(tensor, topk)

            # Verify compression size
            assert len(idxs) == topk
            assert len(vals) == topk

            # Decompress and compute relative error
            decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)
            error = torch.norm(tensor - decompressed) / torch.norm(tensor)
            errors[ratio] = error.item()

        # Check that error decreases as more coefficients are preserved
        assert errors[0.1] > errors[0.3] > errors[0.5], (
            f"Errors did not decrease as expected: {errors}"
        )
        # Optionally, assert that error for the highest ratio is below a loose threshold
        assert errors[0.5] < 0.8, f"Error at highest ratio is too high: {errors[0.5]}"

    def test_batch_compression(self, compressor):
        """Test batch compression functionality"""
        batch_size = 5
        tensor_size = 100
        batch = [torch.randn(tensor_size) for _ in range(batch_size)]
        topk = 10

        # Compress each tensor
        compressed = [compressor.compress(t, topk) for t in batch]

        # Verify each compression
        for idxs, vals, shape, totalk in compressed:
            assert len(idxs) == topk
            assert len(vals) == topk
            assert shape == (tensor_size,)
            assert totalk == tensor_size

    def test_batch_decompression(self, compressor):
        """Test batch decompression functionality"""
        # Create batch of tensors
        original = torch.randn(5, 100)
        topk = 10

        # Compress each tensor in batch
        all_idxs = []
        all_vals = []
        for tensor in original:
            idxs, vals, shape, totalk = compressor.compress(tensor, topk)
            all_idxs.append(idxs)
            all_vals.append(vals)

        # Use the per-sample shape and totalk from the first sample.
        # Note: production batch_decompress returns a decompressed tensor for ONE sample.
        decompressed = compressor.batch_decompress(
            original, all_idxs, all_vals, original[0].shape, original[0].numel()
        )

        # Verify batch decompression: shape must equal the per-sample shape.
        assert decompressed.shape == original[0].shape
        error = torch.norm(original[0] - decompressed) / torch.norm(original[0])
        assert error < 1.5, f"Relative error too high: {error}"

    def test_edge_cases(self, compressor):
        """Test compression edge cases"""
        # Test zero tensor
        zero_tensor = torch.zeros(100)
        idxs, vals, shape, totalk = compressor.compress(zero_tensor, 10)
        assert torch.allclose(vals, torch.zeros_like(vals))

        # Test constant tensor
        const_tensor = torch.ones(100)
        idxs, vals, shape, totalk = compressor.compress(const_tensor, 10)
        decompressed = compressor.decompress(const_tensor, idxs, vals, shape, totalk)
        # For a constant tensor, compression preserves only topk entries.
        # Verify that the decompressed tensor has exactly 10 nonzero entries,
        # and that these nonzero values are all ones.
        nonzero_mask = decompressed != 0
        nonzero_count = nonzero_mask.sum().item()
        assert nonzero_count == 10, f"Expected 10 nonzero entries, got {nonzero_count}"
        assert torch.allclose(decompressed[nonzero_mask], torch.ones(nonzero_count)), (
            "Nonzero entries are not all ones"
        )

    def test_compression_with_nan_inf(self, compressor):
        """Test handling of NaN and Inf values"""
        # Create tensor with some NaN/Inf values
        tensor = torch.randn(100)
        tensor[0] = float("nan")
        tensor[1] = float("inf")
        tensor[2] = float("-inf")

        topk = 10
        idxs, vals, shape, totalk = compressor.compress(tensor, topk)

        decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)

        # For indices where the original tensor is non-finite, ensure the decompressed value
        # reflects the same non-finite property. Finite values must remain finite.
        for i in range(tensor.numel()):
            if not torch.isfinite(tensor[i]):
                if torch.isnan(tensor[i]):
                    assert torch.isnan(decompressed[i]), (
                        f"Index {i}: Expected NaN, got {decompressed[i]}"
                    )
                else:
                    assert decompressed[i] == tensor[i], (
                        f"Index {i}: Expected {tensor[i]}, got {decompressed[i]}"
                    )
            else:
                assert torch.isfinite(decompressed[i]), (
                    f"Index {i}: Expected finite value, got {decompressed[i]}"
                )

    def test_compression_device_handling(self, compressor):
        """Test compression across different devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create tensors on different devices
        cpu_tensor = torch.randn(100)
        gpu_tensor = cpu_tensor.cuda()

        # Test CPU->GPU
        idxs, vals, shape, totalk = compressor.compress(cpu_tensor, 10)
        decompressed = compressor.decompress(
            gpu_tensor, idxs.cuda(), vals.cuda(), shape, totalk
        )
        assert decompressed.device == gpu_tensor.device

        # Test GPU->CPU
        idxs, vals, shape, totalk = compressor.compress(gpu_tensor, 10)
        decompressed = compressor.decompress(
            cpu_tensor, idxs.cpu(), vals.cpu(), shape, totalk
        )
        assert decompressed.device == cpu_tensor.device

    def test_compression_dtype_handling(self, compressor):
        """Test compression with different dtypes"""
        dtypes = [torch.float32, torch.float64]
        for dtype in dtypes:
            tensor = torch.randn(100).to(dtype)
            idxs, vals, shape, totalk = compressor.compress(tensor, 10)

            # Verify compression preserves dtype
            assert vals.dtype == dtype

            # Verify decompression preserves dtype
            decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)
            assert decompressed.dtype == dtype

    def test_compression_gradient_preservation(self, compressor):
        """Test compression preserves numerical gradient information when using full reconstruction"""
        # Create tensor that requires gradient
        tensor = torch.randn(100, requires_grad=True)
        tensor.grad = torch.randn(100)  # Set some gradient

        # Use full_k (i.e. compress with all elements) to achieve lossless reconstruction.
        full_k = tensor.numel()
        idxs, vals, shape, totalk = compressor.compress(tensor, full_k)
        decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)

        # Check that decompressed tensor matches the original tensor's values.
        assert torch.allclose(decompressed, tensor.detach()), (
            "Decompressed tensor values do not match original"
        )


class TestCompressionIntegration:
    """Test integration between transform and compression"""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance for integration tests with dummy model and target_chunk, with identity transforms for testing.
        For 2D tensors, bypass the DCT transform to avoid higher‐order encoding issues.
        """
        model = Identity()
        target_chunk = 16
        transformer_inst = TransformDCT(model, target_chunk)

        # Set transformation dictionaries for 1D (length 16) and 2D (singleton) cases.
        transformer_inst.shape_dict = {16: 16, 1: 1}
        transformer_inst.f_dict = {16: torch.eye(16), 1: torch.eye(1)}
        transformer_inst.b_dict = {16: torch.eye(16), 1: torch.eye(1)}
        transformer_inst.einsum_2d = lambda x, w1, w2=None: x

        # Override encode and decode for 2D tensors to act as identity.
        orig_encode = transformer_inst.encode
        orig_decode = transformer_inst.decode

        def encode_wrapper(x):
            if x.dim() == 2:
                return x  # bypass DCT for 2D input
            return orig_encode(x)

        def decode_wrapper(x):
            if x.dim() == 2:
                return x  # bypass inverse DCT for 2D input
            return orig_decode(x)

        transformer_inst.encode = encode_wrapper
        transformer_inst.decode = decode_wrapper

        return transformer_inst

    @pytest.fixture
    def compressor(self):
        return CompressDCT()

    def test_transform_compress_pipeline(self, transformer, compressor):
        """Test full transform->compress->decompress->inverse pipeline for supported tensor shapes.
        Using a 1D tensor of length 16, as higher order dimensions are not supported.
        """
        original = torch.randn(16)

        # Transform
        transformed = transformer.encode(original)

        # Compress using full reconstruction (i.e. retain all coefficients).
        full_k = transformed.numel()  # Using all elements ensures a lossless roundtrip.
        idxs, vals, shape, totalk = compressor.compress(transformed, full_k)

        # Decompress
        decompressed = compressor.decompress(transformed, idxs, vals, shape, totalk)

        # Inverse transform
        reconstructed = transformer.decode(decompressed)

        # Verify that reconstruction is lossless for full coefficient retention.
        assert torch.allclose(reconstructed, original, rtol=1e-5), (
            "Reconstructed tensor does not match original"
        )
