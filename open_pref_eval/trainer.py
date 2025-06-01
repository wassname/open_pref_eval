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
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from contextlib import contextmanager
from typing import Union, Optional, Dict, Any


@contextmanager
def set_tokenizer_options(tokenizer, **kwargs):
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
        with set_tokenizer_options(self.tokenizer, truncation_side="left"):
            prompt = self.tokenizer.encode_plus(
                batch["prompt"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )["input_ids"]
            out["prompt_ids"] = prompt
            out["prompt_truncated"] = len(prompt) >= self.max_prompt_length
        
        # Calculate max completion length
        max_ans_length = (
            self.max_length - len(prompt)
        )

        # Tokenize completions
        with set_tokenizer_options(self.tokenizer, padding_side="right", truncation_side="right"):
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
                
                # Store without padding - collator will handle padding
                out[key+'_ids'] = ans

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
    ).select_columns(
        ["prompt_ids", "chosen_ids", "rejected_ids", 
         "prompt_truncated", "chosen_truncated", "rejected_truncated"]
    )
    
    # Report truncation statistics
    if verbose and len(tokenized_ds) > 0:
        import numpy as np
        prompt_trunc = np.mean(tokenized_ds['prompt_truncated'])
        chosen_trunc = np.mean(tokenized_ds['chosen_truncated'])
        rejected_trunc = np.mean(tokenized_ds['rejected_truncated'])
        
        if prompt_trunc > 0.2:
            logger.error(f"Prompt rows truncated {prompt_trunc:.2%} > 20%")
        if chosen_trunc > 0.2:
            logger.error(f"Chosen rows truncated {chosen_trunc:.2%} > 20%") 
        if rejected_trunc > 0.2:
            logger.error(f"Rejected rows truncated {rejected_trunc:.2%} > 20%")

        logger.info(f"Truncation rates - Prompt: {prompt_trunc:.2%}, "
                   f"Chosen: {chosen_trunc:.2%}, Rejected: {rejected_trunc:.2%}")
    
    return tokenized_ds

def batch_pad_side(
    encoded_inputs,
    pad_token_id: int,
    side: str,
    max_length: Optional[int]= None,
    return_attention_mask: Optional[bool] = None,
    pad_token_type_id: Optional[int] = None,
) -> dict:
    """
    Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

    Args:
        encoded_inputs:
            Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
        max_length: maximum length of the returned list and optionally padding length (see below).
            Will truncate by taking into account the special tokens.
        side:
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        return_attention_mask:
            (optional) Set to False to avoid returning attention mask (default: set to model specifics)

    modified from https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3672
    """
    # FIXME this seems to be for one row, not a batch, we need to
    required_input = encoded_inputs['input_ids']

    if max_length is None:
        max_length = max([len(ids) for ids in encoded_inputs['input_ids']])

    needs_to_be_padded = len(required_input) != max_length

    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)

    if needs_to_be_padded:
        difference = max_length - len(required_input)

        if side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs['input_ids'] = required_input + [pad_token_id] * difference
        elif side == "left":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            encoded_inputs['input_ids'] = [pad_token_id] * difference + required_input
        else:
            raise ValueError(f"Invalid padding strategy:{side}")
        
    # and as torch tensors
    if return_attention_mask:
        encoded_inputs["attention_mask"] = torch.tensor(encoded_inputs["attention_mask"], dtype=torch.bool)
    if "token_type_ids" in encoded_inputs:
        encoded_inputs["token_type_ids"] = torch.tensor(encoded_inputs["token_type_ids"], dtype=torch.long)
    if "special_tokens_mask" in encoded_inputs:
        encoded_inputs["special_tokens_mask"] = torch.tensor(encoded_inputs["special_tokens_mask"], dtype=torch.bool)
    encoded_inputs['input_ids'] = torch.tensor(encoded_inputs['input_ids'], dtype=torch.long)

    return encoded_inputs

@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """Data collator for preference learning that handles prompt-completion pairs.
    
    Tokenizes and pads prompts (left-padded) and completions (right-padded) separately,
    ensuring EOS tokens are properly added and truncation is tracked.
    """
    pad_token_id: int
    eos_token_id: int
    pad_token_type_id: Optional[int] = 0

    def __call__(self, raw_features: list[dict]) -> dict[str, torch.Tensor]:
        """Process a batch of preference examples into model inputs.
        
        Args:
            raw_features: List of dicts with 'prompt', 'chosen', 'rejected' keys
            
        Returns:
            Dict with tokenized and padded inputs for prompts, chosen, and rejected completions
        """
        prompt_ids = [f["prompt_ids"] for f in raw_features]
        chosen_ids = [f["chosen_ids"] for f in raw_features]
        rejected_ids = [f["rejected_ids"] for f in raw_features]

        # 1) Tokenize prompts: left-pad to handle variable lengths
        prompt_batch = batch_pad_side(
            {'input_ids': prompt_ids},
            pad_token_id=self.pad_token_id,
            pad_token_type_id=self.pad_token_type_id,
            return_attention_mask=True,
            side="left",
        )

        # 2) Tokenize completions: right-truncate to leave room for EOS token

        # Append EOS token to each completion
        chosen_inputs_with_eos = [
            ids + [self.eos_token_id] for ids in chosen_ids
        ]
        rejected_inputs_with_eos = [
            ids + [self.eos_token_id] for ids in rejected_ids
        ]

        # 3) Pad completions to max_completion_length (no further truncation)
        max_completion_length = max(
            max(len(ids) for ids in chosen_inputs_with_eos),
            max(len(ids) for ids in rejected_inputs_with_eos)
        )
        chosen_batch = batch_pad_side(
            {"input_ids": chosen_inputs_with_eos},
            max_length=max_completion_length,
            pad_token_id=self.pad_token_id,
            pad_token_type_id=self.pad_token_type_id,
            return_attention_mask=True,
            side="right",
        )
        rejected_batch = batch_pad_side(
            {"input_ids": rejected_inputs_with_eos},
            max_length=max_completion_length,
            pad_token_id=self.pad_token_id,
            pad_token_type_id=self.pad_token_type_id,
            return_attention_mask=True,
            side="right",
        )

        return {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
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
