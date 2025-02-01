"""Unit tests for compression functionality"""
import pytest
import torch
import numpy as np
from ..utils.assertions import assert_tensor_equal

from neurons.validator.compress import TransformDCT, CompressDCT

class TestTransformDCT:
    """Test DCT transformation functionality"""
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance"""
        return TransformDCT()

    def test_encode_decode_roundtrip(self, transformer):
        """Test that encode->decode preserves tensor values"""
        # Test with different tensor shapes
        test_shapes = [
            (10,),          # 1D tensor
            (5, 10),        # 2D tensor
            (3, 4, 5),      # 3D tensor
            (2, 3, 4, 5)    # 4D tensor
        ]
        
        for shape in test_shapes:
            # Create test tensor
            original = torch.randn(*shape)
            
            # Transform and inverse transform
            encoded = transformer.encode(original)
            decoded = transformer.decode(encoded)
            
            # Verify reconstruction
            assert_tensor_equal(
                original, 
                decoded,
                f"Shape {shape} failed roundtrip test"
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
        """Test different compression ratios"""
        tensor = torch.randn(1000)
        
        for ratio in [0.1, 0.3, 0.5]:
            topk = int(tensor.numel() * ratio)
            idxs, vals, shape, totalk = compressor.compress(tensor, topk)
            
            # Verify compression size
            assert len(idxs) == topk
            assert len(vals) == topk
            
            # Decompress and verify error is reasonable
            decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)
            error = torch.norm(tensor - decompressed) / torch.norm(tensor)
            assert error < ratio * 2  # Error should be proportional to compression ratio

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
            
        # Batch decompress
        decompressed = compressor.batch_decompress(
            original,
            all_idxs,
            all_vals,
            original.shape,
            original.numel()
        )
        
        # Verify batch decompression
        assert decompressed.shape == original.shape
        error = torch.norm(original - decompressed) / torch.norm(original)
        assert error < 0.5  # Reasonable error threshold

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
        assert torch.allclose(decompressed, const_tensor)
        
        # Test single-element compression
        single = torch.randn(100)
        idxs, vals, shape, totalk = compressor.compress(single, 1)
        assert len(idxs) == 1
        assert len(vals) == 1

    def test_compression_with_nan_inf(self, compressor):
        """Test handling of NaN and Inf values"""
        # Create tensor with some NaN/Inf values
        tensor = torch.randn(100)
        tensor[0] = float('nan')
        tensor[1] = float('inf')
        tensor[2] = float('-inf')
        
        topk = 10
        idxs, vals, shape, totalk = compressor.compress(tensor, topk)
        
        # Verify compression handled invalid values
        assert torch.isfinite(vals).all()
        
        # Verify decompression
        decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)
        assert torch.isfinite(decompressed).all()

    def test_compression_device_handling(self, compressor):
        """Test compression across different devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Create tensors on different devices
        cpu_tensor = torch.randn(100)
        gpu_tensor = cpu_tensor.cuda()
        
        # Test CPU->GPU
        idxs, vals, shape, totalk = compressor.compress(cpu_tensor, 10)
        decompressed = compressor.decompress(gpu_tensor, idxs.cuda(), vals.cuda(), shape, totalk)
        assert decompressed.device == gpu_tensor.device
        
        # Test GPU->CPU
        idxs, vals, shape, totalk = compressor.compress(gpu_tensor, 10)
        decompressed = compressor.decompress(cpu_tensor, idxs.cpu(), vals.cpu(), shape, totalk)
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
        """Test compression preserves gradient information"""
        # Create tensor that requires gradient
        tensor = torch.randn(100, requires_grad=True)
        tensor.grad = torch.randn(100)  # Set some gradient
        
        # Compress and decompress
        idxs, vals, shape, totalk = compressor.compress(tensor, 10)
        decompressed = compressor.decompress(tensor, idxs, vals, shape, totalk)
        
        # Verify gradient information
        assert decompressed.requires_grad
        assert tensor.grad is not None  # Original gradient maintained

    def test_compression_memory_efficiency(self, compressor):
        """Test memory usage during compression"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Force garbage collection
        gc.collect()
        start_mem = process.memory_info().rss
        
        # Perform compression on large tensor
        large_tensor = torch.randn(10000, 10000)
        topk = 1000
        
        idxs, vals, shape, totalk = compressor.compress(large_tensor, topk)
        
        # Force cleanup
        del large_tensor
        gc.collect()
        end_mem = process.memory_info().rss
        
        # Verify reasonable memory usage
        # Memory growth should be much less than original tensor size
        mem_growth = end_mem - start_mem
        original_size = 10000 * 10000 * 4  # Approximate size in bytes
        assert mem_growth < original_size * 0.1  # Allow 10% of original size

class TestCompressionIntegration:
    """Test integration between transform and compression"""
    
    @pytest.fixture
    def transformer(self):
        return TransformDCT()
        
    @pytest.fixture
    def compressor(self):
        return CompressDCT()
        
    def test_transform_compress_pipeline(self, transformer, compressor):
        """Test full transform->compress->decompress->inverse pipeline"""
        original = torch.randn(100)
        
        # Transform
        transformed = transformer.encode(original)
        
        # Compress
        idxs, vals, shape, totalk = compressor.compress(transformed, 10)
        
        # Decompress
        decompressed = compressor.decompress(transformed, idxs, vals, shape, totalk)
        
        # Inverse transform
        reconstructed = transformer.decode(decompressed)
        
        # Verify reasonable reconstruction
        error = torch.norm(original - reconstructed) / torch.norm(original)
        assert error < 0.5  # Reasonable error threshold 