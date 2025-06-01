from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from datasets import Dataset
import warnings


@contextmanager
def tok_settings(tokenizer, **kwargs):
    """Temporarily modify tokenizer settings."""
    original = {}
    for key, value in kwargs.items():
        if hasattr(tokenizer, key):
            original[key] = getattr(tokenizer, key)
            setattr(tokenizer, key, value)
    try:
        yield tokenizer
    finally:
        for key, value in original.items():
            setattr(tokenizer, key, value)


@dataclass
class PreTokenizer:
    """Pre-tokenizes preference data with truncation tracking."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    max_prompt_length: int = 128

    def __call__(self, batch: Dict[str, str]) -> Dict[str, Any]:
        """Tokenize preference data row.
        
        Args:
            batch: Row with 'prompt', 'chosen', 'rejected' keys
            
        Returns:
            Tokenized data with truncation flags and masks
        """
        out = {}

        # Encode prompt with left truncation
        with tok_settings(self.tokenizer, truncation_side="left"):
            prompt = self.tokenizer.encode_plus(
                batch["prompt"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )["input_ids"]
        
        # Calculate max completion length
        max_ans_length = (
            self.max_length - len(prompt) - self.tokenizer.num_special_tokens_to_add()
        )

        # Tokenize completions
        with tok_settings(self.tokenizer, padding_side="right", truncation_side="right"):
            for key in ["chosen", "rejected"]:
                # Tokenize completion
                ans = self.tokenizer.encode_plus(
                    batch[key],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_ans_length,
                )["input_ids"]
                
                # Track truncation
                out[key + "_truncated"] = len(ans) >= max_ans_length

                # Combine prompt + completion + special tokens
                ids = prompt + ans
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
                
                # Store without padding - collator will handle padding
                out[key] = ids

        # Create prompt mask for loss computation (without padding)
        out["prompt_truncated"] = len(prompt) >= self.max_prompt_length
        prompt_with_bos = self.tokenizer.build_inputs_with_special_tokens(prompt)[:-1]
        out["prompt_mask"] = [1] * len(prompt_with_bos)

        return out


def tokenize_dataset(dataset, tokenizer: PreTrainedTokenizerBase, 
                    max_length: int = 512, max_prompt_length: int = 128,
                    batch_size: int = 1000, verbose: bool = False):
    """Pre-tokenize a preference dataset with memory-efficient batching.
    
    Args:
        dataset: HuggingFace dataset with 'prompt', 'chosen', 'rejected' columns
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        batch_size: Batch size for processing (to avoid OOM)
        verbose: Whether to report truncation statistics
        
    Returns:
        Tokenized dataset with added columns for tokenized data and truncation flags
    """
    pre_tokenizer = PreTokenizer(
        tokenizer=tokenizer,
        max_length=max_length, 
        max_prompt_length=max_prompt_length
    )
    
    # Process in batches to avoid OOM
    logger.info(f"Tokenizing dataset with {len(dataset)} examples in batches of {batch_size}")
    tokenized_ds = dataset.map(
        pre_tokenizer,
        batched=False,
        batch_size=batch_size,
        desc="Tokenizing"
    )
    
    # Report truncation statistics
    if verbose and len(tokenized_ds) > 0:
        import numpy as np
        prompt_trunc = np.mean(tokenized_ds['prompt_truncated'])
        chosen_trunc = np.mean(tokenized_ds['chosen_truncated'])
        rejected_trunc = np.mean(tokenized_ds['rejected_truncated'])
        
        if prompt_trunc > 0.2:
            logger.error(f"Prompt truncated {prompt_trunc:.2%} > 20%")
        if chosen_trunc > 0.2:
            logger.error(f"Chosen truncated {chosen_trunc:.2%} > 20%") 
        if rejected_trunc > 0.2:
            logger.error(f"Rejected truncated {rejected_trunc:.2%} > 20%")
            
        logger.info(f"Truncation rates - Prompt: {prompt_trunc:.2%}, "
                   f"Chosen: {chosen_trunc:.2%}, Rejected: {rejected_trunc:.2%}")
    
    return tokenized_ds


@dataclass  
class DataCollatorForPreference(DataCollatorMixin):
    """Data collator for pre-tokenized preference data.
    
    Handles batching of pre-tokenized data with proper padding and tensor conversion.
    For on-the-fly tokenization, use PreTokenizer first to tokenize the dataset.
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate pre-tokenized preference data into batches with dynamic padding.
        
        Args:
            features: List of pre-tokenized samples with keys:
                - chosen: tokenized chosen completion (no padding)
                - rejected: tokenized rejected completion (no padding)
                - prompt_mask: mask for prompt tokens (for loss computation, no padding)
            
        Returns:
            Batched tensors with proper padding ready for model forward pass
        """
        # Extract raw token sequences
        chosen_sequences = [f["chosen"] for f in features]
        rejected_sequences = [f["rejected"] for f in features]
        prompt_masks = [f["prompt_mask"] for f in features]
        
        # Find max lengths for padding
        max_chosen_len = max(len(seq) for seq in chosen_sequences)
        max_rejected_len = max(len(seq) for seq in rejected_sequences)
        
        # Pad sequences to batch max length
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        chosen_ids = []
        chosen_masks = []
        rejected_ids = []
        rejected_masks = []
        padded_prompt_masks = []
        
        for chosen, rejected, prompt_mask in zip(chosen_sequences, rejected_sequences, prompt_masks):
            # Pad chosen
            chosen_padded = chosen + [pad_token_id] * (max_chosen_len - len(chosen))
            chosen_mask = [1] * len(chosen) + [0] * (max_chosen_len - len(chosen))
            chosen_ids.append(chosen_padded)
            chosen_masks.append(chosen_mask)
            
            # Pad rejected
            rejected_padded = rejected + [pad_token_id] * (max_rejected_len - len(rejected))
            rejected_mask = [1] * len(rejected) + [0] * (max_rejected_len - len(rejected))
            rejected_ids.append(rejected_padded)
            rejected_masks.append(rejected_mask)
            
            # Pad prompt mask
            prompt_mask_padded = prompt_mask + [0] * (max_chosen_len - len(prompt_mask))
            padded_prompt_masks.append(prompt_mask_padded)

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_masks, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long), 
            "rejected_attention_mask": torch.tensor(rejected_masks, dtype=torch.long),
            "prompt_mask": torch.tensor(padded_prompt_masks, dtype=torch.long),
        }


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


def simple_forward(
    model: nn.Module,
    batch: dict[str, torch.LongTensor],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Tensor]:
    """Run model forward pass on chosen/rejected full sequences.
    
    Args:
        model: The language model to run inference on
        batch: Batch of tokenized inputs with full sequences (prompt + completion)
        device: Target device (auto-detected if None)
        dtype: Target dtype (auto-detected if None)
        
    Returns:
        Dict containing log probabilities and other metrics for chosen/rejected
    """
    # Auto-detect device and dtype from model if not provided
    if device is None:
        device = str(next(model.parameters()).device)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    
    # Move tensors to target device
    chosen_input_ids = batch["chosen_input_ids"].to(device)
    rejected_input_ids = batch["rejected_input_ids"].to(device) 
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    prompt_mask = batch["prompt_mask"].to(device)

    # Run forward passes for chosen and rejected
    with torch.autocast(device, dtype):
        chosen_outputs = model(chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False)
        rejected_outputs = model(rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False)
    
    chosen_logits = chosen_outputs.logits
    rejected_logits = rejected_outputs.logits

    # Prepare labels by shifting input_ids (standard language modeling)
    chosen_labels = torch.roll(chosen_input_ids, shifts=-1, dims=1)
    rejected_labels = torch.roll(rejected_input_ids, shifts=-1, dims=1)
    
    # Create completion masks (inverse of prompt mask)
    completion_mask_chosen = (~prompt_mask.bool() & chosen_attention_mask.bool())
    completion_mask_rejected = (~prompt_mask[:rejected_input_ids.shape[0]].bool() & rejected_attention_mask.bool())
    
    # Shift masks to align with labels
    completion_mask_chosen_shifted = torch.roll(completion_mask_chosen, shifts=-1, dims=1) 
    completion_mask_rejected_shifted = torch.roll(completion_mask_rejected, shifts=-1, dims=1)
    
    # Zero out invalid positions in labels
    chosen_labels[~completion_mask_chosen_shifted] = 0
    rejected_labels[~completion_mask_rejected_shifted] = 0
    
    # Compute log probabilities
    chosen_logprobs = chosen_logits.log_softmax(dim=-1)
    rejected_logprobs = rejected_logits.log_softmax(dim=-1)
    
    # Get per-token log probs
    chosen_per_token_logprobs = torch.gather(chosen_logprobs, dim=-1, index=chosen_labels.unsqueeze(-1)).squeeze(-1)
    rejected_per_token_logprobs = torch.gather(rejected_logprobs, dim=-1, index=rejected_labels.unsqueeze(-1)).squeeze(-1)
    
    # Zero out masked positions and align with original sequence
    chosen_per_token_logprobs[~completion_mask_chosen_shifted] = 0
    rejected_per_token_logprobs[~completion_mask_rejected_shifted] = 0
    chosen_per_token_logprobs = torch.roll(chosen_per_token_logprobs, shifts=1, dims=1)
    rejected_per_token_logprobs = torch.roll(rejected_per_token_logprobs, shifts=1, dims=1)

    output = {
        # Input IDs (for debugging/analysis)
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        
        # Model outputs
        "chosen_logits": chosen_logits,
        "rejected_logits": rejected_logits,
        
        # Per-token log probabilities
        "chosen_logps": chosen_per_token_logprobs,
        "rejected_logps": rejected_per_token_logprobs,
        
        # Summary statistics  
        "mean_chosen_logits": chosen_logits[completion_mask_chosen].mean(),
        "mean_rejected_logits": rejected_logits[completion_mask_rejected].mean(),
        
        # Attention masks for loss computation
        "chosen_mask": completion_mask_chosen,
        "rejected_mask": completion_mask_rejected,
    }

    return output


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

    # Split results back into chosen vs rejected
    prompt_length = prompt_input_ids.shape[1]
    
    output = {
        
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
        
        # Summary statistics
        "mean_chosen_logits": logits[:batch_size][loss_mask_shifted[:batch_size]].mean(),
        "mean_rejected_logits": logits[batch_size:][loss_mask_shifted[batch_size:]].mean(),
        
        # Attention masks for loss computation
        "chosen_mask": loss_mask_shifted[:batch_size][:, prompt_length:],
        "rejected_mask": loss_mask_shifted[batch_size:][:, prompt_length:],
    }

    return output


# Test utilities and dummy data
def dummy_dataset(n: int = 10) -> Dataset:
    """Create a small dummy preference dataset for testing."""
    return Dataset.from_dict({
        "prompt": ["Test prompt"] * n,
        "chosen": ["Good response"] * n, 
        "rejected": ["Bad response"] * n,
    })


@dataclass
class OPEConfig:
    """Configuration for open preference evaluation."""
    output_dir: str = "/tmp/ope_test"
    per_device_eval_batch_size: int = 1
    max_length: int = 512
    max_prompt_length: int = 128


class OPETrainer:
    """Simple trainer wrapper for evaluation."""
    
    def __init__(self, model, tokenizer, args: OPEConfig, train_dataset=None, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer  
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset


def get_dummy_trainer(model, tokenizer) -> OPETrainer:
    """Create a dummy trainer for testing."""
    config = OPEConfig()
    return OPETrainer(
        model=model,
        tokenizer=tokenizer, 
        args=config,
        train_dataset=dummy_dataset(),
        eval_dataset=dummy_dataset(),
    )
