from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin



@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    tokenizer: AutoTokenizer
    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int

    def __call__(self, raw_features: list[dict]) -> dict[str, torch.Tensor]:
        prompts = [f["prompt"] for f in raw_features]
        choiceds = [f["chosen"] for f in raw_features]
        rejects = [f["rejected"] for f in raw_features]

        # 1) prompt: left-pad & truncate
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        prompt_batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="longest",
            return_tensors="pt",
            # return_overflowing_tokens=True,
        )


        # 2) chosen + rejected: first truncate to room for EOS
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        comp_max = self.max_completion_length - 1
        tok_chosen = self.tokenizer(
            choiceds,
            add_special_tokens=False,
            truncation=True,
            max_length=comp_max,
            # return_overflowing_tokens=True,
        )
        tok_reject = self.tokenizer(
            rejects,
            add_special_tokens=False,
            truncation=True,
            max_length=comp_max,
            # return_overflowing_tokens=True,
        )
        # append EOS
        chosen_inputs = [
            ids + [self.tokenizer.eos_token_id] for ids in tok_chosen["input_ids"]
        ]
        rejected_inputs = [
            ids + [self.tokenizer.eos_token_id] for ids in tok_reject["input_ids"]
        ]

        # 3) now pad *only* (no truncation arg!) up to max_completion_length
        chosen_batch = self.tokenizer.pad(
            {"input_ids": chosen_inputs},
            max_length=self.max_completion_length,
            padding="longest",
            return_tensors="pt",
        )
        rejected_batch = self.tokenizer.pad(
            {"input_ids": rejected_inputs},
            max_length=self.max_completion_length,
            padding="longest",
            return_tensors="pt",
        )

        prompt_truncation = (prompt_batch['attention_mask'].all(1) * (prompt_batch['input_ids'].shape[1] == self.max_prompt_length)).float()
        chosen_truncation = (chosen_batch['attention_mask'].all(1) * (chosen_batch['input_ids'].shape[1] == self.max_completion_length)).float()
        rejected_truncation = (rejected_batch['attention_mask'].all(1) * (rejected_batch['input_ids'].shape[1] == self.max_completion_length)).float()
        if prompt_truncation.mean()>0:
            logger.debug(
                f"Batch Prompts were truncated to {self.max_prompt_length} tokens for {prompt_truncation.mean().item():.2%} of samples. Consider increasing max_prompt_length."
            )
        if chosen_truncation.mean()>0:
            logger.debug(
                f"Batch Chosen were truncated to {self.max_completion_length} tokens for {chosen_truncation.mean().item():.2%} of samples. Consider increasing max_completion_length."
            )
        if rejected_truncation.mean()>0:
            logger.debug(
                f"Batch Rejected were truncated to {self.max_completion_length} tokens for {rejected_truncation.mean():.2%} of samples. Consider increasing max_completion_length."
            )

        return {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "prompt_truncation": prompt_truncation,
            "chosen_truncation": chosen_truncation,
            "rejected_truncation": rejected_truncation,
        }


def concatenated_inputs(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    # duplicate prompt
    output["prompt_input_ids"] = torch.cat(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
    )
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )
    # Grab the raw chosen/rejected pairs
    chosen_ids = batch["chosen_input_ids"]
    rejected_ids = batch["rejected_input_ids"]
    chosen_mask = batch["chosen_attention_mask"]
    rejected_mask = batch["rejected_attention_mask"]
    # compute a common max‐completion length
    m = max(chosen_ids.size(1), rejected_ids.size(1))

    # right-pad both to length m (pad_token_id=0 here; adjust if yours is different)
    def pad_right(t: torch.Tensor, length: int, pad_val: int):
        pad_amt = length - t.size(1)
        return F.pad(t, (0, pad_amt), value=pad_val)

    chosen_ids = pad_right(chosen_ids, m, pad_val=0)
    rejected_ids = pad_right(rejected_ids, m, pad_val=0)
    chosen_mask = pad_right(chosen_mask, m, pad_val=0)
    rejected_mask = pad_right(rejected_mask, m, pad_val=0)
    # now safe to concat
    output["completion_input_ids"] = torch.cat((chosen_ids, rejected_ids), dim=0)
    output["completion_attention_mask"] = torch.cat((chosen_mask, rejected_mask), dim=0)
    return output


def gather_ranks(logprobs, labels, loss_mask):
    # More memory-efficient rank calculation
    # Only compute ranks for positions where we need them (loss_mask is True)
    batch_size, seq_len, vocab_size = logprobs.shape
    
    # Get label logprobs for comparison
    label_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # Initialize ranks tensor
    per_token_ranks = torch.zeros_like(labels, dtype=torch.long)
    
    # Only compute ranks where loss_mask is True to save memory
    valid_positions = loss_mask.nonzero(as_tuple=False)
    
    if len(valid_positions) > 0:
        # Process in smaller chunks to avoid OOM
        chunk_size = min(1000, len(valid_positions))
        
        for chunk_start in range(0, len(valid_positions), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(valid_positions))
            chunk_positions = valid_positions[chunk_start:chunk_end]
            
            for i in range(len(chunk_positions)):
                batch_idx, seq_idx = chunk_positions[i]
                label_logprob = label_logprobs[batch_idx, seq_idx]
                
                # Count how many tokens have higher logprob (more efficient than full sort)
                position_logprobs = logprobs[batch_idx, seq_idx]
                rank = (position_logprobs > label_logprob).sum().item() + 1
                per_token_ranks[batch_idx, seq_idx] = rank
    
    per_token_ranks[~loss_mask] = 0  # Set rank to 0 for masked tokens
    per_token_ranks = torch.roll(per_token_ranks, shifts=1, dims=1)
    return per_token_ranks

def concatenated_forward(
    model: nn.Module,
    batch: dict[str, Union[list, torch.LongTensor]],
    device=None,
    dtype=None,
) -> dict[str, Float[Tensor, "b t"]]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    num_examples = batch["prompt_input_ids"].shape[0]

    concatenated_batch = concatenated_inputs(batch)

    model_kwargs = {"use_cache": False}

    if device is None:
        device = str(next(model.parameters()).device)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    concatenated_batch = {k: v.to(device) for k, v in concatenated_batch.items()}

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    # Concatenate the prompt and completion inputs
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat(
        (prompt_attention_mask, completion_attention_mask), dim=1
    )
    # Mask the prompt but not the completion for the loss
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
        dim=1,
    )

    model_kwargs["attention_mask"] = attention_mask

    with torch.autocast(device, dtype):
        outputs = model(input_ids, **model_kwargs)
    logits = outputs.logits

    # Offset the logits by one to align with the labels
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    # Compute the log probabilities of the labels
    labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
    logprobs = logits.log_softmax(-1)
    per_token_logps = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    per_token_logps[~loss_mask] = 0
    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

    per_token_ranks = gather_ranks(logprobs, labels, loss_mask)

    output = {}

    # as in Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
    # we are measuring how concentrated the model is for this token, over the vocabulary
    logp_vocab_conc =  torch.logsumexp(2 * logprobs, dim=-1) # same as sum(probs**2) in log space
    output["logp_vocab_conc_c"] = logp_vocab_conc[:num_examples][:, prompt_input_ids.shape[1]:]
    output["logp_vocab_conc_r"] = logp_vocab_conc[num_examples:][:, prompt_input_ids.shape[1:]]
    
    prompt_length = prompt_input_ids.shape[1]
    output['cho_input_ids'] = input_ids[:num_examples][:, prompt_length:]
    output['rej_input_ids'] = input_ids[num_examples:][:, prompt_length:]
    output['attention_mask'] = attention_mask[:, prompt_length:]
    
    output['chosen_logits'] = logits[:num_examples][:, prompt_length:]
    output['rejected_logits'] = logits[num_examples:][:, prompt_length:]
    output["chosen_logps"] = per_token_logps[:num_examples][:, prompt_length:]
    output["rejected_logps"] = per_token_logps[num_examples:][:, prompt_length:]
    output["chosen_ranks"] = per_token_ranks[:num_examples][:, prompt_length:]
    output["rejected_ranks"] = per_token_ranks[num_examples:][:, prompt_length:]

    mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
    mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

    output["mean_chosen_logits"] = mean_chosen_logits
    output["mean_rejected_logits"] = mean_rejected_logits
    output["chosen_mask"] = loss_mask[:num_examples][:, prompt_length:]
    output["rejected_mask"] = loss_mask[num_examples:][:, prompt_length:]

    return output
