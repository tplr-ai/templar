from typing import Literal

import pytest
import torch
import torch.nn as nn

from tplr.compress import (
    TopKCompressor,
    TransformDCT,
    _dct,
    _get_smaller_split,
    _idct,
    pack_12bit_indices,
    unpack_12bit_indices,
)


class TestTopKCompressor:
    """Test TopKCompressor class using actual implementation"""

    @pytest.fixture
    def compress_instance(self) -> TopKCompressor[Literal[False]]:
        """Create TopKCompressor instance"""
        return TopKCompressor(use_quantization=False)

    @pytest.fixture
    def compress_instance_quantized(self) -> TopKCompressor[Literal[True]]:
        """Create TopKCompressor instance with quantization"""
        return TopKCompressor(
            use_quantization=True, quantization_bins=256, quantization_range=6
        )

    def test_compress_produces_int16_indices(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test that compress() produces 12-bit packed indices"""
        # Create test tensor
        x = torch.randn(10, 10)
        topk = 10

        # Compress using actual method
        idx, val, xshape, totalk = compress_instance.compress(x, topk)

        # Verify index format - should be uint8 tensor for 12-bit packed
        assert idx.dtype == torch.uint8, f"Expected uint8 packed data, got {idx.dtype}"
        assert val.shape[-1] == topk
        assert xshape == x.shape
        # totalk is the size of the last dimension after rearranging
        if len(x.shape) == 2:
            assert totalk == x.shape[-1]  # For 2D tensor, it's the last dimension

    def test_compress_with_quantization(
        self, compress_instance_quantized: TopKCompressor[Literal[True]]
    ):
        """Test compression with quantization enabled"""
        x = torch.randn(10, 10)
        topk = 20

        # Compress with quantization
        result = compress_instance_quantized.compress(x, topk)

        # Should return 5-tuple with quantization
        assert len(result) == 5
        idx, val, _, _, qparams = result

        # idx should be uint8 tensor for 12-bit packed format
        assert idx.dtype == torch.uint8
        assert val.dtype == torch.uint8  # Quantized values
        assert qparams is not None
        assert len(qparams) == 5  # shift, scale, offset, lookup, orig_dtype

    def test_decompress_with_12bit_tuple_format(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test that decompress can handle 12-bit packed tuple format"""
        # Setup
        p = torch.zeros(10, 10)
        xshape = (10, 10)
        totalk = 100

        # Create proper 12-bit packed format using the actual packing function
        # Create indices that are within valid range for a 10x10 tensor (even count)
        original_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int64)

        # Pack using the actual function
        idx = pack_12bit_indices(original_indices)

        val = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32
        )

        # Test decompression with packed format
        result = compress_instance.decompress(p, idx, val, xshape, totalk)
        assert result.shape == xshape
        assert result.dtype == p.dtype

    def test_batch_decompress_multiple_12bit_formats(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test batch_decompress with multiple 12-bit packed indices"""
        # Setup
        p = torch.zeros(10, 10)
        xshape = (10, 10)
        totalk = 100

        # Create multiple 12-bit packed indices
        idx1_orig = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        idx2_orig = torch.tensor([[4, 5], [6, 7]], dtype=torch.int64)

        # Pack them using the 12-bit format
        idx1_packed = pack_12bit_indices(idx1_orig)
        idx2_packed = pack_12bit_indices(idx2_orig)

        idx_list = [idx1_packed, idx2_packed]

        val1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        val2 = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float32)
        val_list = [val1, val2]

        # Test batch decompression
        result = compress_instance.batch_decompress(
            p, idx_list, val_list, xshape, totalk
        )
        assert result.shape == xshape
        assert result.dtype == p.dtype

    def test_compress_decompress_round_trip(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test full compress-decompress round trip"""
        x = torch.zeros(10, 10)
        x[0, 0] = 1.0
        x[1, 1] = 2.0
        x[2, 2] = 3.0
        x[3, 3] = 4.0

        topk = 4

        idx, val, xshape, totalk = compress_instance.compress(x, topk)

        # Verify we got the top-k values
        assert idx.dtype == torch.uint8, "Expected uint8 for 12-bit packed indices"
        assert val.shape[-1] == topk

        # Decompress
        p = torch.zeros_like(x)
        result = compress_instance.decompress(p, idx, val, xshape, totalk)

        # Verify shape
        assert result.shape == x.shape

        # Verify the top values were preserved
        assert result.abs().max() > 0, "Decompressed tensor should have non-zero values"

        # The top 4 values should be approximately 4, 3, 2, 1
        top_vals = torch.topk(result.abs().flatten(), k=4).values
        expected_vals = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert torch.allclose(top_vals, expected_vals, atol=1e-5)

    def test_12bit_index_value_range(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test that indices can represent values appropriate for 12-bit range"""
        # Create a large tensor that would have indices beyond 8-bit range
        x = torch.randn(100, 100)  # 10,000 elements
        topk = 100

        # Compress
        idx, val, _, totalk = compress_instance.compress(x, topk)

        # Check that indices are 12-bit packed format
        assert idx.dtype == torch.uint8, "Expected uint8 for 12-bit packed indices"

        # Since idx is packed, we can't directly check max values
        # Instead verify the packing worked correctly
        # Use val.shape since it has the same shape as the original indices
        unpacked = unpack_12bit_indices(idx, val.shape)

        # Verify some indices might be larger than 255 (8-bit max)
        max_idx = unpacked.max().item()
        assert max_idx < 10000, f"Index {max_idx} exceeds tensor size"

        # If tensor is large enough, we should have indices > 255
        if totalk > 256:
            assert unpacked.max() > 255, (
                "Large tensor should have indices beyond 8-bit range"
            )

    def test_batch_decompress_with_norm_options(
        self, compress_instance: TopKCompressor[Literal[False]]
    ):
        """Test batch_decompress with normalisation and clip_norm options"""
        p = torch.zeros(10, 10)
        xshape = (10, 10)
        totalk = 100

        # Create test data with 12-bit packed format
        idx_orig = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)  # Even count
        idx_packed = pack_12bit_indices(idx_orig)
        idx = [idx_packed]
        val = [torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)]

        # Test with normalisation
        result_norm = compress_instance.batch_decompress(
            p, idx, val, xshape, totalk, normalise=True
        )
        assert result_norm.shape == xshape

        # Test with clip_norm
        result_clip = compress_instance.batch_decompress(
            p, idx, val, xshape, totalk, clip_norm=True
        )
        assert result_clip.shape == xshape

        # Test with block_norms provided
        block_norms = torch.tensor([15.0])
        result_with_norms = compress_instance.batch_decompress(
            p, idx, val, xshape, totalk, block_norms=block_norms, clip_norm=True
        )
        assert result_with_norms.shape == xshape


class TestTransformDCT:
    """Test TransformDCT using actual implementation"""

    @pytest.fixture
    def mock_model(self):
        """Create a simple model for testing"""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(32, 64)
                self.layer2 = nn.Linear(64, 128)

        return SimpleModel()

    def test_transform_init(self, mock_model):
        """Test TransformDCT initialization with real model"""

        target_chunk = 16
        transform = TransformDCT(mock_model, target_chunk)

        # Check that dictionaries were populated
        assert len(transform.shape_dict) > 0
        assert len(transform.f_dict) > 0
        assert len(transform.b_dict) > 0

        # Check that shape_dict contains parameter dimensions
        for param in mock_model.parameters():
            if param.requires_grad:
                for shape_dim in param.shape:
                    assert shape_dim in transform.shape_dict

    def test_encode_decode_real_tensors(self, mock_model):
        """Test encoding and decoding with real model tensors"""
        target_chunk = 8
        transform = TransformDCT(mock_model, target_chunk)

        # Get actual parameter from model
        param = next(mock_model.parameters())

        # Test encoding
        encoded = transform.encode(param, use_dct=False)
        assert encoded.numel() == param.numel()

        # Test decoding
        decoded = transform.decode(encoded, use_dct=False)
        assert decoded.shape == param.shape
        assert torch.allclose(decoded, param.reshape(decoded.shape))

        # Test with DCT
        encoded_dct = transform.encode(param, use_dct=True)
        decoded_dct = transform.decode(encoded_dct, use_dct=True)
        assert decoded_dct.shape == param.shape


class TestUtilityFunctions:
    """Test utility functions using actual implementations"""

    def test_dct_idct_round_trip(self):
        """Test DCT and IDCT implementations"""
        x = torch.randn(4, 8)

        # Apply DCT then IDCT
        X = _dct(x, norm="ortho")
        x_reconstructed = _idct(X, norm="ortho")

        # Should reconstruct original
        assert torch.allclose(x, x_reconstructed, atol=1e-6)

    def test_get_smaller_split(self):
        """Test _get_smaller_split function"""
        # Test with actual use case
        assert _get_smaller_split(64, 8) == 8  # Exact divisor
        assert _get_smaller_split(64, 7) == 4  # Next smaller divisor
        assert _get_smaller_split(64, 100) == 64  # Larger than all divisors
        assert _get_smaller_split(100, 16) == 10  # Find divisor of 100 close to 16


class Test12BitPackingFunctions:
    """Test the pack_12bit_indices and unpack_12bit_indices functions directly"""

    def test_pack_unpack_basic(self):
        """Test basic packing and unpacking preserves values"""
        # Test with simple even-length tensor
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)
        assert unpacked.dtype == torch.int64
        assert packed.dtype == torch.uint8
        # 6 indices = 3 pairs, each pair needs 3 bytes
        assert packed.numel() == 9

    def test_pack_unpack_2d(self):
        """Test packing/unpacking preserves 2D tensor shape"""
        indices = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)
        assert unpacked.shape == indices.shape
        # 8 indices = 4 pairs * 3 bytes = 12 bytes
        assert packed.numel() == 12

    def test_pack_unpack_max_12bit_value(self):
        """Test packing/unpacking with maximum 12-bit value (4095)"""
        indices = torch.tensor([4095, 4094, 0, 1], dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)

    def test_pack_unpack_large_indices(self):
        """Test with indices that span beyond 8-bit range"""
        indices = torch.tensor([0, 255, 256, 1000, 2000, 4000], dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)

    def test_pack_fails_with_odd_count(self):
        """Test that packing fails with odd number of indices"""
        indices = torch.tensor([1, 2, 3], dtype=torch.int64)

        with pytest.raises(ValueError, match="Number of indices must be even"):
            pack_12bit_indices(indices)

    def test_pack_fails_with_value_over_12bit(self):
        """Test that packing fails with values exceeding 12-bit limit"""
        indices = torch.tensor([0, 4096], dtype=torch.int64)  # 4096 > 4095

        with pytest.raises(ValueError, match="exceeds 12-bit limit"):
            pack_12bit_indices(indices)

    def test_unpack_fails_with_odd_count(self):
        """Test that unpacking fails with odd count in shape"""
        packed = torch.zeros(6, dtype=torch.uint8)  # Dummy packed data
        shape = (3,)  # Odd count

        with pytest.raises(ValueError, match="Number of indices must be even"):
            unpack_12bit_indices(packed, shape)

    def test_pack_unpack_empty(self):
        """Test packing/unpacking empty tensor"""
        indices = torch.tensor([], dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)
        assert packed.numel() == 0
        assert unpacked.numel() == 0

    def test_pack_unpack_random_values(self):
        """Test with random valid indices"""
        torch.manual_seed(42)
        # Generate 100 random indices in valid range
        indices = torch.randint(0, 4096, (100,), dtype=torch.int64)

        packed = pack_12bit_indices(indices)
        unpacked = unpack_12bit_indices(packed, indices.shape)

        assert torch.equal(indices, unpacked)
        # 100 indices = 50 pairs * 3 bytes = 150 bytes
        assert packed.numel() == 150

    def test_pack_preserves_device(self):
        """Test that packing preserves tensor device"""
        if torch.cuda.is_available():
            indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")

            packed = pack_12bit_indices(indices)
            unpacked = unpack_12bit_indices(packed, indices.shape)

            assert packed.device == indices.device
            assert unpacked.device == indices.device
            assert torch.equal(indices, unpacked)

    def test_pack_unpack_efficiency(self):
        """Verify that 12-bit packing uses 75% of int16 storage"""
        num_indices = 1000
        indices = torch.arange(num_indices, dtype=torch.int64)

        packed = pack_12bit_indices(indices)

        int16_bytes = num_indices * 2  # 2 bytes per int16
        packed_bytes = packed.numel()  # Each uint8 is 1 byte

        efficiency = packed_bytes / int16_bytes
        assert abs(efficiency - 0.75) < 0.01, (
            f"Expected 75% efficiency, got {efficiency:.2%}"
        )


class Test12BitPackingVerification:
    """Verify 12-bit packing characteristics without implementing the packing"""

    def test_12bit_storage_efficiency(self):
        """Verify that 12-bit packing would be more efficient than int16"""
        # Calculate theoretical storage requirements
        num_values = 1000

        # int16 storage
        int16_bytes = num_values * 2  # 2 bytes per value

        # 12-bit storage (3 bytes per 2 values)
        packed_pairs = num_values // 2
        odd_value = num_values % 2
        packed_bytes = packed_pairs * 3 + (odd_value * 2)

        # Verify 12-bit is more efficient
        efficiency = packed_bytes / int16_bytes
        assert efficiency == 0.75, (
            f"12-bit packing should use 75% of int16 storage, got {efficiency:.2%}"
        )

    def test_12bit_value_limits(self):
        """Verify 12-bit value range"""
        max_12bit = 2**12 - 1  # 4095
        max_int16 = 2**15 - 1  # 32767 (signed)

        assert max_12bit == 4095
        assert max_12bit < max_int16

        # Verify a tensor with indices up to 4095 can be represented
        large_tensor_size = 5000
        assert max_12bit < large_tensor_size, "12-bit can handle indices up to 4095"

        # But 12-bit cannot handle very large tensors
        very_large_tensor = 10000
        assert very_large_tensor > max_12bit, "12-bit limited for very large tensors"
