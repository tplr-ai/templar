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

import tplr
from tplr.model_factory import initialize_torchtitan_model


class TitanLlamaLM(LM):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        # lm_eval expects these properties
        self._batch_size = 1  # Can be adjusted
        self._max_length = 2048  # Max sequence length for the model
        self.eot_token_id = self.tokenizer.eos_token_id
        self.prefix_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

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
        res = []
        for inst in tqdm(requests, desc="loglikelihood", disable=disable_tqdm):
            context_str, continuation_str = inst.args
            context_enc = self.tok_encode(context_str)
            continuation_enc = self.tok_encode(continuation_str)

            # Truncate from the left to ensure the sequence fits within the model's max length.
            # The continuation part must be preserved.
            max_context_len = self._max_length - len(continuation_enc)
            if len(context_enc) > max_context_len:
                context_enc = context_enc[-max_context_len:]
            
            # Prepare inputs
            input_ids = torch.tensor(
                context_enc + continuation_enc, dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Get logits
            with torch.autocast(
                device_type=self.device.split(":")[0], dtype=torch.bfloat16
            ):
                logits = self.model(input_ids)

            # Calculate log probabilities for the continuation
            # Shift logits and labels to align for loss calculation
            # The continuation starts after the context
            continuation_start_idx = len(context_enc)

            # Logits for the continuation tokens
            continuation_logits = logits[
                :, continuation_start_idx - 1 : -1, :
            ]  # -1 because we predict the next token

            # Target tokens for the continuation
            continuation_target_ids = torch.tensor(
                continuation_enc, dtype=torch.long, device=self.device
            )

            # Flatten for F.cross_entropy
            continuation_logits = continuation_logits.view(-1, logits.shape[-1])
            continuation_target_ids = continuation_target_ids.view(-1)

            # Calculate log probabilities
            log_probs = F.log_softmax(continuation_logits, dim=-1)
            log_prob = log_probs.gather(1, continuation_target_ids.unsqueeze(-1)).sum().item()

            # Check if greedy
            # For greedy, we need to check if the argmax of the logits matches the continuation tokens
            greedy_tokens = continuation_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == continuation_target_ids).all().item()

            res.append((log_prob, is_greedy))
        return res

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
        "--actual_batch_size", type=int, default=1, help="Evaluation batch size."
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
        model=model, tokenizer=hparams.tokenizer, device=device
    )

    # 5. Run simple_evaluate only on the master rank
    if rank == 0:
        from lm_eval.tasks import get_task_dict
        task_dict = get_task_dict(args.tasks.split(","))

        results = simple_evaluate(
            model=lm_eval_model,
            tasks=task_dict,
            num_fewshot=args.num_fewshot,
            batch_size=args.actual_batch_size,
        )
        print(results)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
