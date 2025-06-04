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



# def concatenated_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
#     """Concatenate chosen and rejected inputs for efficient batch processing.
    
#     Args:
#         batch: Dict containing prompt, chosen, and rejected input tensors
        
#     Returns:
#         Dict with concatenated tensors for model forward pass
#     """
#     output: dict[str, torch.Tensor] = {}
    
#     # Duplicate prompts to match chosen + rejected structure
#     output["prompt_input_ids"] = torch.cat(
#         [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
#     )
#     output["prompt_attention_mask"] = torch.cat(
#         [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
#     )
    
#     # Extract chosen/rejected completion data
#     chosen_ids = batch["chosen_input_ids"]
#     rejected_ids = batch["rejected_input_ids"]
#     chosen_mask = batch["chosen_attention_mask"]
#     rejected_mask = batch["rejected_attention_mask"]
    
#     # Find common max completion length
#     max_completion_length = max(chosen_ids.size(1), rejected_ids.size(1))

#     def pad_right(tensor: torch.Tensor, target_length: int, pad_value: int = 0) -> torch.Tensor:
#         """Right-pad tensor to target length."""
#         pad_amount = target_length - tensor.size(1)
#         return F.pad(tensor, (0, pad_amount), value=pad_value) if pad_amount > 0 else tensor

#     # Pad both to same length and concatenate
#     chosen_ids_padded = pad_right(chosen_ids, max_completion_length)
#     rejected_ids_padded = pad_right(rejected_ids, max_completion_length)
#     chosen_mask_padded = pad_right(chosen_mask, max_completion_length)
#     rejected_mask_padded = pad_right(rejected_mask, max_completion_length)
    
#     output["completion_input_ids"] = torch.cat((chosen_ids_padded, rejected_ids_padded), dim=0)
#     output["completion_attention_mask"] = torch.cat((chosen_mask_padded, rejected_mask_padded), dim=0)
    
#     return output


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

    prompt_attention_mask = batch["prompt_attention_mask"]
    output = {
        "prompt_attention_mask": prompt_attention_mask,
    }
    for key in ["chosen", "rejected"]:
        # Create loss mask: ignore prompt tokens, only compute loss on completion tokens
        input_ids = batch[f"{key}_input_ids"]
        attention_mask = batch[f"{key}_attention_mask"]
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), attention_mask), dim=1
        )

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
            f"{key}_logprobs": per_token_logprobs,
            f"{key}_mask": loss_mask_shifted,
            f"{key}_input_ids": input_ids,
            f"{key}_attention_mask": attention_mask,
            f"{key}_hidden_states": hs,
            f"{key}_special_tokens_mask": batch.get(f"{key}_special_tokens_mask", None),
        })

    return output
