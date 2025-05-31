from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
import warnings


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """Data collator for preference learning that handles prompt-completion pairs.
    
    Tokenizes and pads prompts (left-padded) and completions (right-padded) separately,
    ensuring EOS tokens are properly added and truncation is tracked.
    """
    tokenizer: AutoTokenizer
    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int

    def __call__(self, raw_features: list[dict]) -> dict[str, torch.Tensor]:
        """Process a batch of preference examples into model inputs.
        
        Args:
            raw_features: List of dicts with 'prompt', 'chosen', 'rejected' keys
            
        Returns:
            Dict with tokenized and padded inputs for prompts, chosen, and rejected completions
        """
        prompts = [f["prompt"] for f in raw_features]
        chosen_completions = [f["chosen"] for f in raw_features]
        rejected_completions = [f["rejected"] for f in raw_features]

        # 1) Tokenize prompts: left-pad & truncate to handle variable lengths
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        prompt_batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="longest",
            return_tensors="pt",
        )

        # 2) Tokenize completions: right-truncate to leave room for EOS token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        completion_max_length = self.max_completion_length - 1  # Reserve space for EOS
        
        tokenized_chosen = self.tokenizer(
            chosen_completions,
            add_special_tokens=False,
            truncation=True,
            max_length=completion_max_length,
        )
        tokenized_rejected = self.tokenizer(
            rejected_completions,
            add_special_tokens=False,
            truncation=True,
            max_length=completion_max_length,
        )
        
        # Append EOS token to each completion
        chosen_inputs_with_eos = [
            ids + [self.tokenizer.eos_token_id] for ids in tokenized_chosen["input_ids"]
        ]
        rejected_inputs_with_eos = [
            ids + [self.tokenizer.eos_token_id] for ids in tokenized_rejected["input_ids"]
        ]

        # 3) Pad completions to max_completion_length (no further truncation)
        chosen_batch = self.tokenizer.pad(
            {"input_ids": chosen_inputs_with_eos},
            max_length=self.max_completion_length,
            padding="longest",
            return_tensors="pt",
        )
        rejected_batch = self.tokenizer.pad(
            {"input_ids": rejected_inputs_with_eos},
            max_length=self.max_completion_length,
            padding="longest",
            return_tensors="pt",
        )

        # Track truncation statistics for monitoring
        prompt_truncation = self._calculate_truncation_rate(prompt_batch, self.max_prompt_length)
        chosen_truncation = self._calculate_truncation_rate(chosen_batch, self.max_completion_length)
        rejected_truncation = self._calculate_truncation_rate(rejected_batch, self.max_completion_length)
        
        self._log_truncation_warnings(prompt_truncation, chosen_truncation, rejected_truncation)

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

    def _calculate_truncation_rate(self, batch: dict, max_length: int) -> torch.Tensor:
        """Calculate the fraction of sequences that were truncated."""
        attention_mask = batch['attention_mask']
        sequence_lengths = attention_mask.sum(dim=1)
        was_truncated = (sequence_lengths == max_length).float()
        return was_truncated
    
    def _log_truncation_warnings(self, prompt_trunc: torch.Tensor, chosen_trunc: torch.Tensor, rejected_trunc: torch.Tensor):
        """Log warnings if significant truncation is detected."""
        if ( prompt_trunc.mean() > 0) or (chosen_trunc.mean() > 0) or (rejected_trunc.mean() > 0):
            warnings.warn("Truncation detected in some sequences. Consider adjusting max lengths.")


def concatenated_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate chosen and rejected inputs for efficient batch processing.
    
    Args:
        batch: Dict containing prompt, chosen, and rejected input tensors
        
    Returns:
        Dict with concatenated tensors for model forward pass
    """
    output: dict[str, torch.Tensor] = {}
    
    # Duplicate prompts to match chosen + rejected structure
    output["prompt_input_ids"] = torch.cat(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
    )
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )
    
    # Extract chosen/rejected completion data
    chosen_ids = batch["chosen_input_ids"]
    rejected_ids = batch["rejected_input_ids"]
    chosen_mask = batch["chosen_attention_mask"]
    rejected_mask = batch["rejected_attention_mask"]
    
    # Find common max completion length
    max_completion_length = max(chosen_ids.size(1), rejected_ids.size(1))

    def pad_right(tensor: torch.Tensor, target_length: int, pad_value: int = 0) -> torch.Tensor:
        """Right-pad tensor to target length."""
        pad_amount = target_length - tensor.size(1)
        return F.pad(tensor, (0, pad_amount), value=pad_value) if pad_amount > 0 else tensor

    # Pad both to same length and concatenate
    chosen_ids_padded = pad_right(chosen_ids, max_completion_length)
    rejected_ids_padded = pad_right(rejected_ids, max_completion_length)
    chosen_mask_padded = pad_right(chosen_mask, max_completion_length)
    rejected_mask_padded = pad_right(rejected_mask, max_completion_length)
    
    output["completion_input_ids"] = torch.cat((chosen_ids_padded, rejected_ids_padded), dim=0)
    output["completion_attention_mask"] = torch.cat((chosen_mask_padded, rejected_mask_padded), dim=0)
    
    return output


def compute_token_ranks_efficient(
    logprobs: torch.Tensor, 
    labels: torch.Tensor, 
    loss_mask: torch.Tensor,
    chunk_size: int = 1000
) -> torch.Tensor:
    """Efficiently compute token ranks with memory optimization.
    
    Args:
        logprobs: Token log probabilities [batch_size, seq_len, vocab_size]
        labels: True token labels [batch_size, seq_len]  
        loss_mask: Boolean mask for valid positions [batch_size, seq_len]
        chunk_size: Processing chunk size to avoid OOM
        
    Returns:
        Token ranks tensor [batch_size, seq_len]
    """
    batch_size, seq_len, vocab_size = logprobs.shape
    
    # Get log probabilities for the actual labels
    label_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # Initialize ranks tensor
    token_ranks = torch.zeros_like(labels, dtype=torch.long)
    
    # Only compute ranks where loss_mask is True
    valid_positions = loss_mask.nonzero(as_tuple=False)
    
    if len(valid_positions) == 0:
        return torch.roll(token_ranks, shifts=1, dims=1)
    
    # Process in chunks to avoid memory issues
    for chunk_start in range(0, len(valid_positions), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(valid_positions))
        chunk_positions = valid_positions[chunk_start:chunk_end]
        
        for pos_idx in range(len(chunk_positions)):
            batch_idx, seq_idx = chunk_positions[pos_idx]
            label_logprob = label_logprobs[batch_idx, seq_idx]
            
            # Count tokens with higher log probability (rank = number of better tokens + 1)
            position_logprobs = logprobs[batch_idx, seq_idx]
            rank = (position_logprobs > label_logprob).sum().item() + 1
            token_ranks[batch_idx, seq_idx] = rank
    
    # Mask out invalid positions and shift by 1 (standard practice)
    token_ranks[~loss_mask] = 0
    return torch.roll(token_ranks, shifts=1, dims=1)


def concatenated_forward(
    model: nn.Module,
    batch: dict[str, Union[list, torch.LongTensor]],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Tensor]:
    """Run model forward pass on concatenated chosen/rejected inputs.

    This approach concatenates chosen and rejected completions to avoid 
    doing two separate forward passes, which is more efficient for FSDP.
    
    Args:
        model: The language model to run inference on
        batch: Batch of tokenized inputs from DataCollatorForPreference
        device: Target device (auto-detected if None)
        dtype: Target dtype (auto-detected if None)
        
    Returns:
        Dict containing log probabilities, ranks, and other metrics for chosen/rejected
    """
    batch_size = batch["prompt_input_ids"].shape[0]

    # Prepare concatenated inputs
    concatenated_batch = concatenated_inputs(batch)

    # Auto-detect device and dtype from model if not provided
    if device is None:
        device = str(next(model.parameters()).device)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    
    # Move tensors to target device
    concatenated_batch = {k: v.to(device) for k, v in concatenated_batch.items()}

    # Extract concatenated components
    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    # Create full sequences by concatenating prompts and completions
    full_input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    full_attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    
    # Create loss mask: ignore prompt tokens, only compute loss on completion tokens
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
    )

    # Run model forward pass with mixed precision
    with torch.autocast(device, dtype):
        outputs = model(full_input_ids, attention_mask=full_attention_mask, use_cache=False)
    logits = outputs.logits

    # Prepare labels by shifting input_ids (standard language modeling)
    labels = torch.roll(full_input_ids, shifts=-1, dims=1)
    loss_mask_shifted = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    # Mask out invalid label positions
    labels[~loss_mask_shifted] = 0  # dummy token for masked positions
    
    # Compute log probabilities and per-token log probs
    logprobs = logits.log_softmax(dim=-1)
    per_token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    per_token_logprobs[~loss_mask_shifted] = 0  # Zero out masked positions
    per_token_logprobs = torch.roll(per_token_logprobs, shifts=1, dims=1)  # Align with original sequence

    # Compute token ranks efficiently
    token_ranks = compute_token_ranks_efficient(logprobs, labels, loss_mask_shifted)

    # Calculate vocabulary concentration (WPO paper Eq. 2)
    # Measures how concentrated the model's distribution is over the vocabulary
    vocab_concentration = torch.logsumexp(2 * logprobs, dim=-1)  # log(sum(probs^2))

    # Split results back into chosen vs rejected
    prompt_length = prompt_input_ids.shape[1]
    
    output = {
        # Vocabulary concentration measures  
        "vocab_concentration_chosen": vocab_concentration[:batch_size][:, prompt_length:],
        "vocab_concentration_rejected": vocab_concentration[batch_size:][:, prompt_length:],
        
        # Input IDs (for debugging/analysis)
        "chosen_input_ids": full_input_ids[:batch_size][:, prompt_length:],
        "rejected_input_ids": full_input_ids[batch_size:][:, prompt_length:],
        "attention_mask": full_attention_mask[:, prompt_length:],
        
        # Model outputs
        "chosen_logits": logits[:batch_size][:, prompt_length:],
        "rejected_logits": logits[batch_size:][:, prompt_length:],
        
        # Per-token log probabilities
        "chosen_logps": per_token_logprobs[:batch_size][:, prompt_length:],
        "rejected_logps": per_token_logprobs[batch_size:][:, prompt_length:],
        
        # Token ranks
        "chosen_ranks": token_ranks[:batch_size][:, prompt_length:],
        "rejected_ranks": token_ranks[batch_size:][:, prompt_length:],
        
        # Summary statistics
        "mean_chosen_logits": logits[:batch_size][loss_mask_shifted[:batch_size]].mean(),
        "mean_rejected_logits": logits[batch_size:][loss_mask_shifted[batch_size:]].mean(),
        
        # Attention masks for loss computation
        "chosen_mask": loss_mask_shifted[:batch_size][:, prompt_length:],
        "rejected_mask": loss_mask_shifted[batch_size:][:, prompt_length:],
    }

    return output
