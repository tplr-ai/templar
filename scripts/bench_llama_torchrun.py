#ruff: noqa
#!/usr/bin/env python3
"""
Benchmark 4096×32 Llama on four optimisation variants – *sequentially on one GPU*:

1. base            (no compile, no flash-attn)
2. compile         (torch.compile, default mode)
3. flash           (flash-attention-2)
4. compile+flash   (both, default mode)

Key Change: Uses forward-only warmup for compiled models to potentially avoid CUDA graph errors.

Prerequisites:
  - Install necessary libraries: torch, transformers, tabulate, flash-attn
  - Ensure CUDA is working for the target GPU.

Launch Command Example (run on GPU 0):
    python scripts/bench_llama_sequential.py --gpu 0
"""

# ── env setup ──
import os
import json
import time
import gc
import argparse
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM
from tabulate import tabulate
import torch.compiler
import torch._dynamo # For reset

torch.set_float32_matmul_precision("high")

# ───────── hyper-params ──────────────────────────────────────────────────────
DTYPE, SEQ_LEN, BATCH = torch.bfloat16, 2048, 1
SLACK_GB, REPEATS     = 4, 2

# Configuration (same as before)
BASE_CFG = dict( hidden_size=4096, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336, max_position_embeddings=8192, hidden_act="silu", rms_norm_eps=1e-5, vocab_size=128256, tie_word_embeddings=False, use_cache=False, torch_dtype=DTYPE)
VARIANTS = { "base": {}, "compile": {"compile": True}, "flash": {"flash": True}, "compile+flash": {"compile": True, "flash": True}}
ORDER = ["base", "compile", "flash", "compile+flash"]

# ───────── utils ─────────────────────────────────────────────────────────────
def _peak_gb(dev) -> float:
    if torch.cuda.is_available() and dev.index < torch.cuda.device_count(): return torch.cuda.max_memory_allocated(dev) / 1.073742e9
    return 0.0

def bench(variant: str, target_gpu_id: int) -> dict:
    dev = torch.device(f"cuda:{target_gpu_id}")
    torch.cuda.set_device(dev)
    print(f"Running variant '{variant}' on {dev}")
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats(dev)

    total_gb, usable_gb = 0, 0
    try: total_gb = torch.cuda.get_device_properties(dev).total_memory / 1.073742e9; usable_gb = total_gb - SLACK_GB; print(f"Device Memory: Total={total_gb:.2f} GB, Usable (Est.)={usable_gb:.2f} GB")
    except Exception as e: print(f"Warning: Error getting GPU properties: {e}")

    cfg_d = BASE_CFG.copy()
    use_flash = VARIANTS[variant].get("flash", False)
    if use_flash:
        try: import flash_attn; cfg_d["_attn_implementation"] = "flash_attention_2"; print("Using flash_attention_2.")
        except ImportError: print("WARNING - flash-attention-2 requested but not found/installed. Falling back.")
        if "_attn_implementation" in cfg_d: del cfg_d["_attn_implementation"]
    else: print("Using default attention implementation.")

    try: cfg = LlamaConfig(**cfg_d)
    except Exception as e: print(f"FATAL - Failed to create LlamaConfig: {e}"); return {"name": variant, "gpu": target_gpu_id, "error": f"Config creation failed: {e}"}

    # --- Model Loading ---
    load_s = 0.0; model = None
    try:
        t0 = time.perf_counter(); print("Loading model...")
        model = LlamaForCausalLM(cfg).to(dtype=DTYPE).to(dev) # Use .to(dtype)
        if hasattr(model, 'supports_gradient_checkpointing') and model.supports_gradient_checkpointing: model.gradient_checkpointing_enable(); print("Gradient checkpointing enabled.")
        elif hasattr(model, 'gradient_checkpointing_enable'):
             try: model.gradient_checkpointing_enable(); print("Gradient checkpointing enabled (method found).")
             except Exception as gc_e: print(f"WARNING - Failed to enable GC via method: {gc_e}")
        else: print("WARNING - Model does not appear to support gradient checkpointing.")
        load_s = time.perf_counter() - t0; print(f"Model loaded in {load_s:.2f}s.")
    except torch.cuda.OutOfMemoryError:
        print("FATAL - CUDA Out Of Memory during model loading!")
        peak_gb_at_oom = _peak_gb(dev); del model; model = None; torch.cuda.empty_cache(); gc.collect()
        return {"name": variant, "gpu": target_gpu_id, "error": "OOM during model load", "peak_gb": peak_gb_at_oom, "load_s": load_s, "usable_gb": usable_gb}
    except Exception as e: print(f"FATAL - Failed to load model: {e}"); import traceback; traceback.print_exc(); return {"name": variant, "gpu": target_gpu_id, "error": f"Model load failed: {e}", "load_s": load_s}

    # --- Compilation (if requested) ---
    compile_s = 0.0; is_compiled = False
    if model is not None and VARIANTS[variant].get("compile", False):
        if hasattr(torch, 'compile'):
            compile_mode = "max-autotune"
            torch._dynamo.reset()
            print(f"Compiling model (mode='{compile_mode}', dynamic=False)...")
            t0c = time.perf_counter()
            try:
                model_compiled = torch.compile(model, mode=compile_mode, dynamic=False)
                model = model_compiled; is_compiled = True

                print("Running dummy pass to trigger compilation...")
                dummy_input = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN), device=dev)
                # Use FWD-only for dummy pass too, mirroring the new warmup
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    _ = model(dummy_input)
                torch.cuda.synchronize(dev); compile_s = time.perf_counter() - t0c
                print(f"Compilation finished in {compile_s:.2f}s.")
                del dummy_input; torch.cuda.empty_cache(); gc.collect() # Removed loss/backward here
            except torch.cuda.OutOfMemoryError:
                 print("ERROR - CUDA Out Of Memory during model compilation/dummy pass!")
                 compile_s = time.perf_counter() - t0c; is_compiled = False
                 print("Reloading original model due to compile OOM..."); del model; torch.cuda.empty_cache(); gc.collect()
                 try:
                     model = LlamaForCausalLM(cfg).to(dtype=DTYPE).to(dev)
                     if hasattr(model, 'gradient_checkpointing_enable'): model.gradient_checkpointing_enable()
                     print("Original model reloaded. Continuing uncompiled.")
                 except Exception as reload_e: print(f"FATAL - Failed to reload: {reload_e}"); return {"name": variant, "gpu": target_gpu_id, "error": "Compile OOM + reload failed", "compile_s": compile_s, "load_s": load_s}
            except Exception as e: print(f"ERROR during model compilation: {e}"); import traceback; traceback.print_exc(); compile_s = 0.0; is_compiled = False; print("Continuing benchmark with uncompiled model.")
        else: print("WARNING - torch.compile not available. Skipping compilation.")

    # --- Benchmarking Loop ---
    torch.cuda.reset_peak_memory_stats(dev); times = []; oom_occurred = False; error_msg_detail = None
    if REPEATS >= 1 and model is not None:
        print("Running warmup iteration...")
        try:
            x = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN), device=dev)
            # *** MODIFIED WARMUP ***
            if is_compiled:
                print("Warmup (Compiled - Forward only)...")
                # No need for mark_step_begin before no_grad typically, graph runs FWD
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    _ = model(x) # Only forward pass
            else:
                print("Warmup (Not Compiled - Forward + Backward)...")
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    outputs = model(x); loss = outputs.logits.to(torch.float32).mean(); loss.backward() # Full FWD+BWD
                del outputs, loss # Clean up intermediates from FWD+BWD warmup
            # *** END MODIFIED WARMUP ***

            torch.cuda.synchronize(dev); del x; torch.cuda.empty_cache()
            print("Warmup done.")
        except torch.cuda.OutOfMemoryError: print("FATAL - OOM during warmup!"); oom_occurred = True; error_msg_detail = "OOM during warmup"
        except RuntimeError as e: print(f"FATAL - RUNTIME ERROR during warmup: {e}"); import traceback; traceback.print_exc(); oom_occurred = True; error_msg_detail = f"Runtime Error warmup: {e}"
        except Exception as e: print(f"FATAL - UNEXPECTED ERROR during warmup: {e}"); import traceback; traceback.print_exc(); oom_occurred = True; error_msg_detail = f"Error warmup: {e}"

    if not oom_occurred and model is not None: torch.cuda.reset_peak_memory_stats(dev)
    elif model is not None: print("Skipping benchmark runs due to failure during warmup.")

    if not oom_occurred and model is not None:
        print(f"Starting benchmark runs ({REPEATS} repeats)...")
        for i in range(REPEATS):
            try:
                # Mark step is needed before the actual FWD+BWD timed runs for compiled models
                if is_compiled: torch.compiler.cudagraph_mark_step_begin()
                x = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN), device=dev)
                torch.cuda.synchronize(dev); t0 = time.perf_counter()
                with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
                    outputs = model(x); loss = outputs.logits.to(torch.float32).mean(); loss.backward()
                torch.cuda.synchronize(dev); iter_time = time.perf_counter() - t0; times.append(iter_time)
                print(f"Rep {i+1}/{REPEATS} done ({iter_time*1000:.1f} ms).")
                del x, loss, outputs; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError: print(f"FATAL - OOM during benchmark iter {i+1}!"); oom_occurred = True; error_msg_detail = f"OOM after {len(times)} reps"; break
            except RuntimeError as e: print(f"FATAL - RUNTIME ERROR during benchmark iter {i+1}: {e}"); import traceback; traceback.print_exc(); oom_occurred = True; error_msg_detail = f"Runtime Error after {len(times)} reps: {e}"; break
            except Exception as e: print(f"FATAL - UNEXPECTED ERROR during benchmark iter {i+1}: {e}"); import traceback; traceback.print_exc(); oom_occurred = True; error_msg_detail = f"Unexpected Error after {len(times)} reps: {e}"; break

    # --- Result Calculation & Reporting (Identical to previous version) ---
    peak_gb = _peak_gb(dev); n_params = 0; param_gb = 0
    if model is not None: n_params = sum(p.numel() for p in model.parameters() if p.requires_grad); bytes_per_param = 2 if DTYPE in [torch.bfloat16, torch.float16] else 4; param_gb = n_params * bytes_per_param / 1.073742e9
    else:
        if not error_msg_detail: error_msg_detail = "Model loading failed"
    avg_time_s = (sum(times) / len(times)) if times else 0; tok_s = (BATCH * SEQ_LEN) / avg_time_s if avg_time_s > 0 else 0
    final_error_msg = error_msg_detail if oom_occurred or error_msg_detail else None
    res = dict( name=variant, gpu=target_gpu_id, error=final_error_msg, params_M=n_params / 1e6, peak_gb=peak_gb, param_gb=param_gb, act_gb=max(0, peak_gb - param_gb), tok_s=tok_s if not final_error_msg else 0, t_ms=(1000 * avg_time_s) if not final_error_msg else 0, compile_s=compile_s, load_s=load_s, usable_gb=usable_gb, total_gb=total_gb, headroom_gb=max(0, usable_gb - peak_gb), repeats_completed=len(times) )
    if final_error_msg: print(f"Benchmark FAILED ({final_error_msg}). Final Peak Memory: {peak_gb:.2f} GB.")
    else: print(f"Benchmark done. Final Peak Memory: {peak_gb:.2f} GB. Avg time: {res['t_ms']:.1f} ms. Tok/s: {tok_s:.0f}")
    print("Cleaning up model and cache for this variant..."); del model; torch.cuda.empty_cache(); gc.collect()
    return res

# ───────── main ──────────────────────────────────────────────────────────────
# (Main function remains identical to previous sequential script version)
def main(args):
    target_gpu_id = args.gpu
    if not torch.cuda.is_available(): print("ERROR: CUDA is not available."); return
    if target_gpu_id >= torch.cuda.device_count(): print(f"ERROR: GPU ID {target_gpu_id} invalid. Available: {torch.cuda.device_count()}"); return
    print(f"Starting sequential benchmarks on GPU {target_gpu_id}...")
    results = []
    for variant in ORDER:
        print(f"\n{'='*15} Running Variant: {variant} {'='*15}")
        try: result = bench(variant=variant, target_gpu_id=target_gpu_id); results.append(result)
        except Exception as e: print(f"CRITICAL ERROR during execution of bench() for variant {variant}: {e}"); import traceback; traceback.print_exc(); results.append({"name": variant, "gpu": target_gpu_id, "error": f"Critical bench error: {e}"})
        finally: print("--- Cleaning up GPU memory between variants ---"); torch.cuda.empty_cache(); gc.collect() # Ensure cleanup
    print("\n\n{'='*15} Final Benchmark Results {'='*15}")
    if results:
        hdr = ["Config", "GPU", "Peak GB", "Param GB", "Act GB", "Tok/s", "Time ms", "Compile s", "Load s", "Headroom GB", "Status (#Reps)"]
        rows = []; results_map = {r.get('name'): r for r in results if r.get('name')}; ordered_results_data = [results_map.get(v_name) for v_name in ORDER] # Use ORDER for map lookup too
        for i, r in enumerate(results): # Iterate through actual results collected
             v_name = r.get("name", ORDER[i] if i < len(ORDER) else "Unknown") # Handle missing name defensively
             if r is None: rows.append([f"{v_name}?", target_gpu_id, "-", "-", "-", "-", "-", "-", "-", "-", "Internal Error"]); continue
             status = "OK"; reps_completed = r.get('repeats_completed', REPEATS if not r.get("error") else 0); reps_info = f"({reps_completed}/{REPEATS})"
             if r.get("error"): status = f"ERROR ({r['error']})"; reps_info = f"({r.get('repeats_completed', 0)}/{REPEATS})"
             rows.append([ r.get("name", "N/A"), r.get("gpu", target_gpu_id), f'{r.get("peak_gb", 0):.1f}' if "peak_gb" in r else "-", f'{r.get("param_gb", 0):.2f}' if "param_gb" in r else "-", f'{r.get("act_gb", 0):.1f}' if "act_gb" in r else "-", f'{r.get("tok_s", 0):.0f}' if "tok_s" in r and status=="OK" else "-", f'{r.get("t_ms", 0):.1f}' if "t_ms" in r and status=="OK" else "-", f'{r.get("compile_s", 0):.1f}' if r.get("compile_s", 0) > 0.01 else "-", f'{r.get("load_s", 0):.1f}' if "load_s" in r else "-", f'{r.get("headroom_gb", 0):.1f}' if "headroom_gb" in r else "-", f"{status} {reps_info}" ])
        try: print(tabulate(rows, headers=hdr, tablefmt="grid"))
        except Exception as tab_e: print(f"Error generating table: {tab_e}\nRaw results:\n{results}")
        print("-----------------------------------------------------\n")
        combined_path = Path("bench_sequential_results.json")
        try: combined_path.write_text(json.dumps(results, indent=2)); print(f"Combined results saved to {combined_path}")
        except Exception as e: print(f"ERROR saving combined results: {e}")
    else: print("No results were collected.")
    print("Sequential benchmark run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Llama benchmarks sequentially on a single GPU.")
    parser.add_argument("--gpu", type=int, default=0, help="The ID of the GPU to use (default: 0).")
    args = parser.parse_args()
    main(args)