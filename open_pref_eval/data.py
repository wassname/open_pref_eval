
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
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

# Use tokenizer.apply_chat_template() to handle chat templates
def apply_chat_template_to_completion(tokenizer, prompt, completion):
    """Apply chat template to a single completion."""
    return tokenizer.apply_chat_template(
        conversation=[{"role":"assistant", "content": f"{prompt.rstrip()+' '}<|split|>{completion}"}],
        add_generation_prompt=False,
        add_special_tokens=True,
        tokenize=False
    ).split('<|split|>')

# https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/falcon-instruct.jinja
FALCAN_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ system_message | trim }}
{% for message in messages %}
    {% set content = message['content'].replace('\r\n', '\n').replace('\n\n', '\n') %}
    {{ '\n\n' + message['role'] | capitalize + ': ' + content | trim }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '\n\nAssistant:' }}
{% endif %}"""


@dataclass
class PreTokenizer:
    """Pre-tokenizes preference data with truncation tracking."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    max_prompt_length: int = 512

    def __call__(self, batch: Dict[str, str]) -> Dict[str, Any]:
        """Tokenize preference data row.
        
        Args:
            batch: Row with 'prompt', 'chosen', 'rejected' keys
            
        Returns:
            Tokenized data with truncation flags and masks


        Note: people do this in various ways for preference data, I do it in the simplest way. Pretokenize, pre-pad, pre-truncate, pre-concat. I find the complexity is not worth the tiny performance gain.
        """
        out = {}
        
        if self.tokenizer.chat_template is None:
            logger.warning(
                "No chat template set for tokenizer, using default FALCAN INSTRUCTR template."
            )
            self.tokenizer.chat_template = FALCAN_CHAT_TEMPLATE
        
        # Apply chat template to prompt and completions
        # TODO handle both being messages
        batch['prompt'], batch['chosen'] = apply_chat_template_to_completion(self.tokenizer, batch["prompt"], batch["chosen"])
        batch['rejected'] = apply_chat_template_to_completion(self.tokenizer, batch["prompt"], batch["rejected"])[1]

        # Encode prompt with left truncation
        prompt = self.tokenizer.encode_plus(
            batch["prompt"],
            add_special_tokens=False,
            padding=False,
        )["input_ids"]
        out["prompt_ids"] = prompt

        # Tokenize completions
        for key in ["chosen", "rejected"]:
            # Tokenize completion
            ans = self.tokenizer.encode_plus(
                batch[key],
                add_special_tokens=False,
            )["input_ids"]
            out[key + "_ids"] = ans

        # Truncation: Now we know the lengths of prompt and completions
        max_completion_length = max(
            len(out["chosen_ids"]),
            len(out["rejected_ids"]),
        )
        prompt_length = len(out["prompt_ids"])
        out['prompt_truncated'] = 0
        out['chosen_truncated'] = 0
        out['rejected_truncated'] = 0
        if (max_completion_length + prompt_length) > self.max_length:
            # first truncate prompt to max_prompt_length
            if prompt_length > self.max_prompt_length:
                out["prompt_truncated"] = prompt_length - self.max_prompt_length
                prompt = prompt[-self.max_prompt_length:]
                prompt_length = self.max_prompt_length
            
            # then truncate completions to fit
            if (max_completion_length + prompt_length) > self.max_length:
                max_ans_length = self.max_length - prompt_length
                if len(out["chosen_ids"]) > max_ans_length:
                    out["chosen_truncated"] = len(out["chosen_ids"]) - max_ans_length
                    out["chosen_ids"] = out["chosen_ids"][:max_ans_length]
                if len(out["rejected_ids"]) > max_ans_length:
                    out["rejected_truncated"] = len(out["rejected_ids"]) - max_ans_length
                    out["rejected_ids"] = out["rejected_ids"][:max_ans_length]

        # Now join and pad, store prompt mask and attention mask
        out["prompt_mask"] = [1] * len(prompt) + [0] * (self.max_length - len(prompt))
        with set_tokenizer_options(self.tokenizer, padding_side="right", truncation_side="right"):
            for key in ["chosen", "rejected"]:
                ids = prompt + out[key + "_ids"]

                special_tokens_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

                # pad and attention mask
                encoded_inputs = self.tokenizer.pad(
                    {
                        "input_ids": ids,
                        "special_tokens_mask": special_tokens_mask,
                    },
                    max_length=self.max_length,
                    padding="max_length",
                    return_attention_mask=True,
                )

                out[key + "_ids"] = encoded_inputs["input_ids"]
                out[key + "_mask"] = encoded_inputs["attention_mask"]
                out[key + "_special_tokens_mask"] = encoded_inputs["special_tokens_mask"]

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

    if tokenizer.pad_token_id is None:
        warnings.warn(
            "Tokenizer does not have a pad token set. "
            "Setting pad token to eos token for preference data."
        )
        tokenizer.pad_token = tokenizer.eos_token

    pre_tokenizer = PreTokenizer(
        tokenizer=tokenizer,
        max_length=max_length, 
        max_prompt_length=max_prompt_length
    )
    
    # Process in batches to avoid OOM
    logger.debug(f"Tokenizing dataset with in batches of {batch_size}")
    tokenized_ds = dataset.map(
        pre_tokenizer,
        batched=False,
        batch_size=batch_size,
        desc="Tokenizing"
    ).select_columns(
        ["chosen_ids", "rejected_ids", 
         "prompt_truncated", "chosen_truncated", "rejected_truncated",
         "prompt_mask", "chosen_mask", "rejected_mask",
         "chosen_special_tokens_mask", "rejected_special_tokens_mask",
         ]
    )
    
    # Report truncation statistics    
    ds_qc = tokenized_ds
    if isinstance(tokenized_ds, dict):
        ds_qc = next(iter(tokenized_ds.values()))
    ds_qc = ds_qc.with_format("torch")
    if len(tokenized_ds) > 0:
        prompt_trunc = torch.mean((ds_qc['prompt_truncated']>1)*1.0)
        chosen_trunc = torch.mean((ds_qc['chosen_truncated']>1)*1.0)
        rejected_trunc = torch.mean((ds_qc['rejected_truncated']>1)*1.0)

        if prompt_trunc > 0.2:
            logger.error(f"Prompt rows truncated {prompt_trunc:.2%} > 20%")
        if chosen_trunc > 0.2:
            logger.error(f"Chosen rows truncated {chosen_trunc:.2%} > 20%") 
        if rejected_trunc > 0.2:
            logger.error(f"Rejected rows truncated {rejected_trunc:.2%} > 20%")

    if verbose > 0:
        logger.info(f"Truncation rates - Prompt: {prompt_trunc:.2%}, "
                    f"Chosen: {chosen_trunc:.2%}, Rejected: {rejected_trunc:.2%}")
    if verbose > 1:
        # QC sample decode when verbose
        row = ds_qc[0]
        s = "=== Sample QC after tokenization ==="
        s += "\n" + f"Chosen: {tokenizer.decode(row['chosen_ids'])}"
        s += "\n---"
        s += "\n" + f"Rejected: {tokenizer.decode(row['rejected_ids'])}"
        s += "\n=== End QC sample ==="
        logger.info(s)
    
    return tokenized_ds

