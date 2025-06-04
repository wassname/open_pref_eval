from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
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


def concatenated_forward(
    model: nn.Module,
    batch: dict[str, Union[list, torch.LongTensor]],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any
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

    # Auto-detect device and dtype from model if not provided
    if device is None:
        device = str(next(model.parameters()).device)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    
    # Move tensors to target device
    batch = {k: v.to(device) for k, v in batch.items()}

    prompt_mask = batch["prompt_mask"].bool()
    output = {
        "prompt_mask": prompt_mask,
    }
    for key in ["chosen", "rejected"]:
        # Create loss mask: ignore prompt tokens, only compute loss on completion tokens
        input_ids = batch[f"{key}_ids"]
        attention_mask = batch[f"{key}_mask"]
        loss_mask = attention_mask & ~prompt_mask  # Only compute loss on completion tokens

        # Run model forward pass with mixed precision
        with torch.autocast(device, dtype):
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
        logits = outputs.logits

        # Prepare labels by shifting input_ids (standard language modeling)
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask_shifted = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        # Mask out invalid label positions
        labels[~loss_mask_shifted] = 0  # dummy token for masked positions
        
        # Compute log probabilities and per-token log probs
        logprobs = logits.log_softmax(dim=-1)
        per_token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        per_token_logprobs[~loss_mask_shifted] = 0  # Zero out masked positions
        per_token_logprobs = torch.roll(per_token_logprobs, shifts=1, dims=1)  # Align with original sequence

        hs = outputs.hidden_states if outputs.hidden_states is not None else None

        output.update({
            f"{key}_logits": logits,
            f"{key}_logps": per_token_logprobs,
            f"{key}_mask": loss_mask_shifted,
            f"{key}_ids": input_ids,
            f"{key}_attention_mask": attention_mask,
            f"{key}_hidden_states": hs,
            f"{key}_special_tokens_mask": batch.get(f"{key}_special_tokens_mask", None),
        })

    return output
