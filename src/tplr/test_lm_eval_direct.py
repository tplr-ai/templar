import argparse
import os
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm
from accelerate.utils.memory import find_executable_batch_size

import tplr
from tplr.model_factory import initialize_torchtitan_model


class TitanLlamaLM(LM):
    def __init__(self, model, tokenizer, hparams, device="cuda", actual_batch_size=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.model_size = hparams.model_size

        # lm_eval expects these properties
        self._batch_size_arg = actual_batch_size
        self._batch_size = None
        self._max_length = 2048  # Max sequence length for the model
        self.eot_token_id = self.tokenizer.eos_token_id
        self.prefix_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

    @property
    def batch_size(self):
        if self._batch_size is None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                ideal_batch_size = 128  # Ideal batch size for this model
                if self._batch_size_arg == "auto":
                    self._batch_size = int(ideal_batch_size * 0.9)
                    print(f"Using 'auto' batch size. Set to a safe heuristic: {self._batch_size}")
                elif self._batch_size_arg == "auto_unsafe":
                    self._batch_size = ideal_batch_size
                    print(f"Using 'auto_unsafe' batch size. Set to the full ideal batch size: {self._batch_size}")
                else:
                    try:
                        self._batch_size = int(self._batch_size_arg)
                        print(f"Using specified batch size: {self._batch_size}")
                    except ValueError:
                        print(f"Invalid batch size: {self._batch_size_arg}. Defaulting to a safe batch size of {int(ideal_batch_size * 0.9)}")
                        self._batch_size = int(ideal_batch_size * 0.9)
        return self._batch_size

    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def prefix_token_id(self):
        return (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, **kwargs)

    def _model_call(self, inps):
        # This method is typically used by HFLM. We will directly implement _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # This method is typically used by HFLM. We will directly implement generate_until
        raise NotImplementedError()

    @torch.no_grad()
    def loglikelihood(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(0, len(requests), self.batch_size), desc="loglikelihood", disable=disable_tqdm):
                batch = requests[i:i+self.batch_size]
                
                inputs = []
                for inst in batch:
                    if len(inst.args) != 2:
                        continue
                    context_str, continuation_str = inst.args
                    context_enc = self.tok_encode(context_str)
                    continuation_enc = self.tok_encode(continuation_str)

                    max_context_len = self._max_length - len(continuation_enc)
                    if len(context_enc) > max_context_len:
                        context_enc = context_enc[-max_context_len:]
                    
                    inputs.append({
                        "input_ids": torch.tensor(context_enc + continuation_enc, dtype=torch.long),
                        "continuation_enc": torch.tensor(continuation_enc, dtype=torch.long),
                        "context_len": len(context_enc)
                    })

                if not inputs:
                    continue

                # Pad the batch
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    [d["input_ids"] for d in inputs], batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                logits = self.model(padded_inputs)

                for j, data in enumerate(inputs):
                    continuation_start_idx = data["context_len"]
                    continuation_end_idx = continuation_start_idx + len(data["continuation_enc"])
                    continuation_logits = logits[j, continuation_start_idx - 1 : continuation_end_idx - 1, :]
                    continuation_target_ids = data["continuation_enc"].to(self.device)

                    continuation_logits = continuation_logits.view(-1, logits.shape[-1])
                    continuation_target_ids = continuation_target_ids.view(-1)

                    log_probs = F.log_softmax(continuation_logits, dim=-1)
                    log_prob = log_probs.gather(1, continuation_target_ids.unsqueeze(-1)).sum().item()

                    greedy_tokens = continuation_logits.argmax(dim=-1)
                    is_greedy = (greedy_tokens == continuation_target_ids).all().item()

                    results.append((log_prob, is_greedy))
        return results

    @torch.no_grad()
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        for request in tqdm(requests, desc="generate_until", disable=disable_tqdm):
            context_str = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}

            # Extract generation parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", 50)
            stop = gen_kwargs.get("until", None)

            input_ids = torch.tensor(
                self.tok_encode(context_str), dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Simple greedy decoding loop
            generated_tokens = []
            for _ in range(max_gen_toks):
                with torch.autocast(
                    device_type=self.device.split(":")[0], dtype=torch.bfloat16
                ):
                    logits = self.model(input_ids)

                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            generated_text = self.tok_decode(generated_tokens)

            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break

            res.append(generated_text)
        return res

    @torch.no_grad()
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        log_likelihoods = []

        for instance in tqdm(requests, desc="loglikelihood_rolling", disable=disable_tqdm):
            text = instance.args[0]
            tokens = self.tok_encode(text)
            
            if not tokens:
                log_likelihoods.append(float('nan'))
                continue

            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            # For rolling loglikelihood, we are interested in the log probability of the whole sequence
            with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
                logits = self.model(input_ids)

            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # The log likelihood is the negative of the sum of the losses
            log_likelihood = -loss.sum().item()
            log_likelihoods.append(log_likelihood)
            
        return log_likelihoods


def main():
    # 1. Argument parsing
    parser = argparse.ArgumentParser(
        description="Test script for lm_eval with TitanLlama model."
    )
    parser.add_argument(
        "--netuid", type=int, default=3, help="Bittensor network UID."
    )
    parser.add_argument(
        "--actual_batch_size", type=str, default="auto", help="Evaluation batch size."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for evaluation"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="piqa",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", type=float, default=None, help="Limit the number of examples to evaluate"
    )
    args = parser.parse_args()

    # 2. Initialize distributed environment
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if args.device != "cuda:0" and "cuda" in args.device: # If a specific cuda device is requested
        device = args.device
    elif args.device == "cpu":
        device = "cpu"


    # 3. Load hparams and initialize the TitanLlama model
    hparams = tplr.load_hparams()
    model = initialize_torchtitan_model(
        hparams=hparams, role="evaluator", device=device, world_size=world_size
    )

    # 4. Create an instance of the custom TitanLlamaLM wrapper
    lm_eval_model = TitanLlamaLM(
        model=model, tokenizer=hparams.tokenizer, hparams=hparams, device=device, actual_batch_size=args.actual_batch_size
    )

    # 5. Run simple_evaluate only on the master rank
    if rank == 0:
        results = simple_evaluate(
            model=lm_eval_model,
            tasks=args.tasks.split(','),
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=lm_eval_model.batch_size,
            device=device,
        )
        print(results)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
