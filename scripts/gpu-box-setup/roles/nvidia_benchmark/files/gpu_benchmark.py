import json
import subprocess
import threading
import time
import platform
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset


# ------------------------------------------------------------------
# Helper: Initialize CUDA contexts to avoid cuBLAS warning
# ------------------------------------------------------------------
def init_cuda_contexts():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        # Creating a dummy tensor forces a CUDA context on each device.
        _ = torch.tensor([0], device=f"cuda:{i}")
        torch.cuda.synchronize()


# ------------------------------------------------------------------
# 1. Memory Usage Benchmark
# ------------------------------------------------------------------
def track_memory_usage(device, model_sizes, batch_size):
    results = {}
    from torchvision.models import resnet18, resnet50

    for size in model_sizes:
        if size == "small":
            model = nn.Sequential(
                nn.Flatten(), nn.Linear(3 * 224 * 224, 256), nn.ReLU(), nn.Linear(256, 10)
            ).to(device)
        elif size == "medium":
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.to(device)
        elif size == "large":
            model = resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.to(device)
        else:
            print(f"Unknown model size '{size}', skipping.")
            continue

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        model.train()

        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        dummy_target = torch.randint(0, 10, (batch_size,), device=device)

        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
        results[size] = peak_memory
        print(f"Model size '{size}': Peak memory usage = {peak_memory:.2f} MB")
    return results


# ------------------------------------------------------------------
# 2. Precision Formats Benchmark
# ------------------------------------------------------------------
def benchmark_precision_formats(device, sizes, num_iters):
    results = {}
    precisions = {
        "FP32": torch.float32,
        "FP16": torch.float16,
    }
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        precisions["BF16"] = torch.bfloat16

    for precision, dtype in precisions.items():
        precision_results = {}
        for M, N, K in sizes:
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)

            # Warm-up
            torch.matmul(a, b)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iters):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / num_iters * 1000  # ms per iteration
            key = f"{M}x{K} * {K}x{N}"
            precision_results[key] = elapsed
            print(f"MatMul {key} ({precision}): {elapsed:.3f} ms per iteration")
        results[precision] = precision_results
    return results


# ------------------------------------------------------------------
# 3. Throughput Benchmark
# ------------------------------------------------------------------
def benchmark_throughput(device, batch_sizes, model_type):
    results = {}
    num_iters = 50
    if model_type.lower() == "resnet50":
        model = models.resnet50(weights=None).to(device)
    else:
        print(f"Unsupported model type: {model_type}")
        return results

    model.eval()
    with torch.no_grad():
        for batch in batch_sizes:
            dummy_input = torch.randn(batch, 3, 224, 224, device=device)
            # Warm-up iterations
            for _ in range(10):
                _ = model(dummy_input)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iters):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            throughput = batch * num_iters / elapsed
            results[batch] = throughput
            print(f"Batch size {batch}: Throughput = {throughput:.2f} images/sec")
    return results


# ------------------------------------------------------------------
# 4. Data Loading Benchmark (skipped for GPU-only mode)
# ------------------------------------------------------------------
def benchmark_data_loading(batch_sizes, num_workers):
    print("Skipping data loading benchmark in GPU-only mode.")
    return {"skipped": "GPU-only mode"}


# ------------------------------------------------------------------
# 5. Thermal Throttling Benchmark
# ------------------------------------------------------------------
def monitor_thermal_throttling(device, duration, test_load):
    results = {"temperatures": []}
    print(f"\nMonitoring GPU temperature for {duration} seconds using {test_load} load.")

    stop_event = threading.Event()

    def stress_test():
        if test_load == "matmul":
            a = torch.randn(4096, 4096, device=device)
            b = torch.randn(4096, 4096, device=device)
            while not stop_event.is_set():
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
        else:
            print(f"Unknown test_load '{test_load}'. No stress test performed.")

    stress_thread = threading.Thread(target=stress_test)
    stress_thread.start()

    start_time = time.time()
    sample_interval = 5  # seconds
    while time.time() - start_time < duration:
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            for line in result.strip().split('\n'):
                temp_str, util_str = line.split(',')
                temp = int(temp_str.strip())
                util = int(util_str.strip())
                results["temperatures"].append({"temp": temp, "util": util})
                print(f"GPU Temperature: {temp} °C, Utilization: {util}%")
                if temp > 85:
                    print("Warning: GPU Temperature exceeds 85°C - potential throttling!")
        except Exception as e:
            print("Failed to query GPU temperature:", e)
        time.sleep(sample_interval)
    stop_event.set()
    stress_thread.join()
    print("Thermal monitoring complete.")
    return results


# ------------------------------------------------------------------
# 6. Memory Bandwidth Benchmark
# ------------------------------------------------------------------
def benchmark_memory_bandwidth(device, sizes, num_iters):
    results = {}
    for size in sizes:
        num_elements = size // 4  # for FP32 (4 bytes)
        src = torch.randn(num_elements, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)
        # Warm-up
        dst.copy_(src)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_time = elapsed / num_iters
        bandwidth = size / avg_time  # bytes/sec
        bandwidth_gb = bandwidth / (1024**3)
        results[str(size)] = {"bandwidth_gb_per_sec": bandwidth_gb, "avg_time_ms": avg_time * 1e3}
        print(
            f"Size {size/1024/1024:.2f} MB: {bandwidth_gb:.2f} GB/s (avg time per copy: {avg_time*1e3:.3f} ms)"
        )
    return results


# ------------------------------------------------------------------
# 7. Convolution Benchmark
# ------------------------------------------------------------------
def benchmark_convolution(device, batch_sizes, num_iters):
    results = {}
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1).to(device)
    conv.eval()

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        # Warm-up
        for _ in range(5):
            _ = conv(dummy_input)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iters):
            _ = conv(dummy_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_time = elapsed / num_iters * 1000  # ms per iteration
        results[batch_size] = avg_time
        print(f"Batch size {batch_size}: Convolution avg time per iteration = {avg_time:.3f} ms")
    return results


# ------------------------------------------------------------------
# 8. Quantized Operations Benchmark (skipped for GPU-only mode)
# ------------------------------------------------------------------
def benchmark_quantized_ops(device, batch_sizes, num_iters):
    print("Skipping quantized operations benchmark in GPU-only mode (CPU-only feature).")
    return {"skipped": "GPU-only mode"}


# ------------------------------------------------------------------
# 9. Transformer Layer Benchmark
# ------------------------------------------------------------------
def benchmark_transformer_layer(device, batch_sizes, seq_lengths, num_iters):
    results = {}
    transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8).to(device)
    transformer_layer.eval()

    for batch in batch_sizes:
        for seq_length in seq_lengths:
            dummy_input = torch.randn(seq_length, batch, 512, device=device)
            for _ in range(5):
                _ = transformer_layer(dummy_input)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_iters):
                _ = transformer_layer(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            avg_time = elapsed / num_iters * 1000  # ms
            key = f"batch_{batch}_seq_{seq_length}"
            results[key] = avg_time
            print(
                f"Batch size {batch}, Seq length {seq_length}: Transformer avg time per iteration = {avg_time:.3f} ms"
            )
    return results


# ------------------------------------------------------------------
# 10. Multi-GPU Scaling Benchmark
# ------------------------------------------------------------------
def benchmark_multi_gpu(batch_sizes, num_iters):
    results = {}
    num_devices = torch.cuda.device_count()
    print("\n[Multi-GPU Scaling Test]")
    print(f"Available GPUs: {num_devices}")
    if num_devices < 2:
        print("Insufficient GPUs for multi-GPU benchmark. Skipping...")
        return {"error": "Insufficient GPUs"}

    target_devices = min(8, num_devices)
    device_ids = list(range(target_devices))
    # Initialize CUDA context for each device to avoid cuBLAS warnings.
    for device_id in device_ids:
        torch.cuda.set_device(device_id)
        torch.tensor([0], device=f"cuda:{device_id}")
        torch.cuda.synchronize()

    model = nn.Linear(1024, 1024).to(f"cuda:{device_ids[0]}")
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 1024, device=f"cuda:{device_ids[0]}")
        for _ in range(5):
            _ = model(dummy_input)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iters):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        throughput = batch_size * num_iters / elapsed
        results[batch_size] = throughput
        print(f"Batch size {batch_size}: Throughput = {throughput:.2f} samples/sec")
    return results


# ------------------------------------------------------------------
# 11. Mixed Precision Training Benchmark (GPU-only)
# ------------------------------------------------------------------
def benchmark_mixed_precision(device, model_type, batch_size, num_iters):
    results = {}
    
    if model_type.lower() == "resnet50":
        model = models.resnet50(weights=None).to(device)
    else:
        print(f"Unsupported model type: {model_type}")
        return {"error": "Unsupported model type"}
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    
    # Input data
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Regular FP32 training benchmark
    model.train()
    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / num_iters * 1000  # ms per iteration
    results["fp32"] = fp32_time
    print(f"FP32 Training: {fp32_time:.3f} ms per iteration")
    
    # Mixed precision training benchmark
    model.train()
    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    amp_time = (time.time() - start) / num_iters * 1000  # ms per iteration
    results["amp"] = amp_time
    results["speedup"] = fp32_time / amp_time
    print(f"Mixed Precision Training: {amp_time:.3f} ms per iteration")
    print(f"Speedup: {results['speedup']:.2f}x")
    
    return results


# ------------------------------------------------------------------
# 12. GPU Memory Profile Benchmark (GPU-only)
# ------------------------------------------------------------------
def profile_gpu_memory_usage():
    results = {}
    
    # Query GPU memory usage with nvidia-smi
    try:
        memory_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        
        gpu_memory = []
        for line in memory_info.strip().split('\n'):
            parts = line.split(',')
            index = int(parts[0].strip())
            used = int(parts[1].strip())
            free = int(parts[2].strip())
            total = int(parts[3].strip())
            
            gpu_memory.append({
                "gpu_index": index,
                "memory_used_mb": used,
                "memory_free_mb": free,
                "memory_total_mb": total,
                "memory_utilization_pct": (used / total) * 100
            })
        
        results["gpu_memory"] = gpu_memory
        
        # Print memory usage info
        print("\nGPU Memory Profile:")
        for gpu in gpu_memory:
            print(f"GPU {gpu['gpu_index']}: {gpu['memory_used_mb']} MB used / {gpu['memory_total_mb']} MB total ({gpu['memory_utilization_pct']:.1f}%)")
        
    except Exception as e:
        print(f"Error profiling GPU memory: {e}")
        results["error"] = str(e)
    
    return results


# ------------------------------------------------------------------
# Main: Run Benchmarks and Save Results
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Run CUDA diagnostic info
    print("=== CUDA Diagnostic Information ===")
    # Check nvidia-smi
    try:
        print("Running nvidia-smi...")
        subprocess.run(["nvidia-smi"], check=False)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
    
    # Check CUDA libraries
    print("\nChecking CUDA library locations...")
    try:
        cuda_paths = subprocess.check_output(["ldconfig", "-p"], text=True)
        print("CUDA libraries found in LD path:")
        for line in cuda_paths.splitlines():
            if "cuda" in line.lower() or "nvidia" in line.lower():
                print(line.strip())
    except Exception as e:
        print(f"Error checking CUDA libraries: {e}")
    
    # Print PyTorch CUDA info
    print("\nPyTorch CUDA Information:")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check if we can proceed
    if not torch.cuda.is_available():
        # Save diagnostic information to a file
        with open("cuda_diagnostics.txt", "w") as f:
            f.write("=== CUDA Diagnostic Information ===\n")
            try:
                nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
                f.write("nvidia-smi output:\n")
                f.write(nvidia_smi + "\n\n")
            except Exception as e:
                f.write(f"Error running nvidia-smi: {e}\n\n")
            
            try:
                env_vars = subprocess.check_output(["env"], text=True)
                f.write("Environment variables:\n")
                for line in env_vars.splitlines():
                    if "cuda" in line.lower() or "nvidia" in line.lower() or "ld_library" in line.lower():
                        f.write(line + "\n")
                f.write("\n")
            except Exception as e:
                f.write(f"Error getting environment: {e}\n\n")
                
            try:
                cuda_paths = subprocess.check_output(["ldconfig", "-p"], text=True)
                f.write("CUDA libraries found in LD path:\n")
                for line in cuda_paths.splitlines():
                    if "cuda" in line.lower() or "nvidia" in line.lower():
                        f.write(line + "\n")
                f.write("\n")
            except Exception as e:
                f.write(f"Error checking CUDA libraries: {e}\n\n")
            
            f.write(f"torch.__version__: {torch.__version__}\n")
            f.write(f"torch.version.cuda: {torch.version.cuda}\n")
            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            
        print("\nDiagnostic information saved to cuda_diagnostics.txt")
        print("Please check if NVIDIA drivers are properly installed and CUDA is set up correctly.")
        print("You can install a CPU-only version of PyTorch to run a basic benchmark.")
        sys.exit(1)
    
    # Capture system information
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "node": platform.node(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    }
    
    # Get and print GPU info
    print(f"GPU Devices: {system_info['gpu_count']}")
    for i, device in enumerate(system_info['gpu_devices']):
        print(f"  {i}: {device}")
    
    device = torch.device("cuda")
    print(f"Running benchmarks on GPU: {torch.cuda.get_device_name(device)}")
    
    # Initialize CUDA contexts to suppress cuBLAS warnings.
    init_cuda_contexts()
    
    # Collect benchmark results.
    benchmark_results = {
        "system_info": system_info,
        "device": device.type,
    }
    
    # Profile initial GPU memory usage
    benchmark_results["initial_memory_profile"] = profile_gpu_memory_usage()
    
    # Run standard benchmarks
    benchmark_results["memory_usage"] = track_memory_usage(device, ["small", "medium", "large"], 32)
    
    benchmark_results["precision_formats"] = benchmark_precision_formats(
        device, sizes=[(1024, 1024, 1024), (2048, 2048, 2048)], num_iters=30
    )
    
    benchmark_results["throughput"] = benchmark_throughput(
        device, batch_sizes=[16, 32, 64, 128], model_type="resnet50"
    )
    
    # Skip data loading benchmark in GPU-only mode
    benchmark_results["data_loading"] = benchmark_data_loading(
        batch_sizes=[32, 64], num_workers=[1, 2, 4]
    )
    
    # For thermal monitoring, run with more intensive load
    benchmark_results["thermal"] = monitor_thermal_throttling(
        device, duration=30, test_load="matmul"
    )
    
    benchmark_results["memory_bandwidth"] = benchmark_memory_bandwidth(
        device, sizes=[10 * 1024 * 1024, 100 * 1024 * 1024, 500 * 1024 * 1024], num_iters=20
    )
    
    benchmark_results["convolution"] = benchmark_convolution(
        device, batch_sizes=[16, 32, 64, 128], num_iters=30
    )
    
    # Skip quantized ops in GPU-only mode
    benchmark_results["quantized_ops"] = benchmark_quantized_ops(
        device, batch_sizes=[16, 32, 64], num_iters=30
    )
    
    benchmark_results["transformer_layer"] = benchmark_transformer_layer(
        device, batch_sizes=[16, 32, 64], seq_lengths=[50, 100, 512], num_iters=30
    )
    
    benchmark_results["multi_gpu"] = benchmark_multi_gpu(
        batch_sizes=[32, 64, 128, 256], num_iters=30
    )
    
    # Add GPU-specific benchmarks
    benchmark_results["mixed_precision"] = benchmark_mixed_precision(
        device, model_type="resnet50", batch_size=64, num_iters=20
    )
    
    # Final memory profile
    benchmark_results["final_memory_profile"] = profile_gpu_memory_usage()

    # Save aggregated results to a JSON file.
    with open("gpu_benchmark_results.json", "w") as f:
        # Convert any non-serializable values to strings
        serializable_results = json.dumps(benchmark_results, default=str)
        json.dump(json.loads(serializable_results), f, indent=4)
        
    print("\nBenchmark results saved to gpu_benchmark_results.json")