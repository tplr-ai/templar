"""Local Model Evaluator

This script evaluates model checkpoints from local files using standardized benchmark tasks.
Instead of pulling checkpoints from the Bittensor network, it accepts checkpoint paths as arguments.

Usage:
    python scripts/evaluator-local.py --checkpoint_path model_checkpoint.pt
    python scripts/evaluator-local.py --checkpoint_path model_checkpoint.pt --tasks arc_challenge,winogrande
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.llama import LlamaForCausalLM

import tplr

MODEL_PATH: str = "models/eval"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local evaluator script for model checkpoints"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate (0.0-1.0)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up model files after evaluation",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[LlamaForCausalLM, dict]:
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")

    hparams = tplr.load_hparams()

    model = LlamaForCausalLM(config=hparams.model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        model_state = checkpoint
        metadata = {}

    model.load_state_dict(model_state)
    model.to(device)  # type: ignore

    tplr.logger.info("Model loaded successfully")
    return model, metadata


def run_evaluation(
    model: LlamaForCausalLM,
    hparams: SimpleNamespace,
    args: argparse.Namespace,
) -> dict:
    """Run lm-eval benchmark and return results."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    hparams.tokenizer.save_pretrained(MODEL_PATH)

    if args.limit < 1.0:
        limit_arg = f"--limit {args.limit}"
    else:
        limit_arg = ""

    if args.num_fewshot > 0:
        fewshot_arg = f"--num_fewshot {args.num_fewshot}"
    else:
        fewshot_arg = ""

    command = f"""
    lm-eval --model hf \
        --model_args pretrained={MODEL_PATH},tokenizer={MODEL_PATH} \
        --tasks {args.tasks} \
        --device {args.device} \
        --batch_size {args.batch_size} \
        --output_path {args.output_dir} \
        {limit_arg} \
        {fewshot_arg}
    """.strip()

    tplr.logger.info(f"Running benchmark: {command}")
    start_time = time.time()
    exit_code = os.system(command)
    runtime = time.time() - start_time

    if exit_code != 0:
        raise RuntimeError(f"Evaluation failed with exit code {exit_code}")

    results_dir = Path(args.output_dir) / "models__eval"
    latest_file = max(results_dir.glob("*.json"), key=os.path.getctime)

    with open(latest_file, "r") as f:
        results = json.load(f)

    if args.cleanup:
        tplr.logger.info("Cleaning up model files")
        shutil.rmtree(MODEL_PATH)
        torch.cuda.empty_cache()
    else:
        tplr.logger.info(f"Model files kept at: {MODEL_PATH}")

    return {
        "benchmark_runtime": runtime,
        "results": results["results"],
        "config": args.__dict__,
    }


def print_results(results: dict) -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nRuntime: {results['benchmark_runtime']:.2f} seconds")
    print(f"Config: {json.dumps(results['config'], indent=2)}")

    print("\nTask Scores:")
    print("-" * 30)

    for task_name, task_results in results["results"].items():
        # Priority order for metrics
        metric_names = ["acc_norm,none", "acc,none"]

        for metric_name in metric_names:
            if (value := task_results.get(metric_name)) is not None:
                print(f"{task_name} ({metric_name}): {value:.4f}")
                break

    print("=" * 50 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        hparams = tplr.load_hparams()
        model, metadata = load_checkpoint(args.checkpoint_path, args.device)

        if metadata:
            tplr.logger.info(f"Checkpoint metadata: {metadata}")

        results = run_evaluation(model, hparams, args)

        print_results(results)

        output_file = Path(args.output_dir) / "evaluation_summary.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        tplr.logger.info(f"Results saved to {output_file}")

    except Exception as e:
        tplr.logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
