from dataclasses import dataclass
from datasets import Dataset
import torch
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from jaxtyping import Float
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import selective_log_softmax, flush_left



dummy_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "[INST] How is the stock price? [/INST]",
        "[INST] How is the stock price? [/INST] ",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Python",
        "$46 as of 10am EST",
        "46 as of 10am EST",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "Java",
        " $46 as of 10am EST",
        " 46 as of 10am EST",
    ],
}
dummy_dataset = Dataset.from_dict(dummy_dataset_dict)



@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    tokenizer: AutoTokenizer
    max_prompt_length:    int
    max_completion_length:int
    pad_token_id:         int
    
    def __call__(self, raw_features: list[dict]) -> dict[str, torch.Tensor]:
        prompts  = [f["prompt"]  for f in raw_features]
        choiceds = [f["chosen"]  for f in raw_features]
        rejects  = [f["rejected"] for f in raw_features]

        # 1) prompt: left-pad & truncate
        self.tokenizer.padding_side = "left"
        prompt_batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="longest",
            return_tensors="pt",
        )

        # 2) chosen + rejected: first truncate to room for EOS
        self.tokenizer.padding_side = "right"
        comp_max = self.max_completion_length - 1
        tok_chosen = self.tokenizer(
            choiceds,
            add_special_tokens=False,
            truncation=True,
            max_length=comp_max,
        )
        tok_reject = self.tokenizer(
            rejects,
            add_special_tokens=False,
            truncation=True,
            max_length=comp_max,
        )
        # append EOS
        chosen_inputs  = [ids + [self.tokenizer.eos_token_id] for ids in tok_chosen["input_ids"]]
        rejected_inputs= [ids + [self.tokenizer.eos_token_id] for ids in tok_reject["input_ids"]]

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

        return {
            "prompt_input_ids":        prompt_batch["input_ids"],
            "prompt_attention_mask":   prompt_batch["attention_mask"],
            "chosen_input_ids":        chosen_batch["input_ids"],
            "chosen_attention_mask":   chosen_batch["attention_mask"],
            "rejected_input_ids":      rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }
    

def concatenated_inputs(
    batch: dict[str, torch.Tensor], 
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}

    # duplicate prompt for chosen/rejected
    output["prompt_input_ids"] = torch.cat(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
    )
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )

    # since chosen_input_ids & rejected_input_ids are already
    # right-padded to the same max_completion_length in the collator,
    # we can just cat them along the batch‐dim:
    output["completion_input_ids"] = torch.cat(
        (batch["chosen_input_ids"], batch["rejected_input_ids"]), dim=0
    )
    output["completion_attention_mask"] = torch.cat(
        (batch["chosen_attention_mask"], batch["rejected_attention_mask"]), dim=0
    )

    return output

def concatenated_forward(model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], device=None, dtype=None) -> dict[str, Float[Tensor, 'b t']]:
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
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    # Mask the prompt but not the completion for the loss
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
        dim=1,
    )

    # Flush left to reduce the memory usage
    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

    model_kwargs["attention_mask"] = attention_mask

    
    with torch.autocast(device, dtype):
        outputs = model(input_ids, **model_kwargs)
    logits = outputs.logits

    # Offset the logits by one to align with the labels
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    # Compute the log probabilities of the labels
    labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
    per_token_logps = selective_log_softmax(logits, labels)
    per_token_logps[~loss_mask] = 0
    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

    output = {}
    output["chosen_logps"] = per_token_logps[:num_examples]
    output["rejected_logps"] = per_token_logps[num_examples:]

    mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
    mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

    output["mean_chosen_logits"] = mean_chosen_logits
    output["mean_rejected_logits"] = mean_rejected_logits
    output["chosen_mask"] = loss_mask[:num_examples]
    output["rejected_mask"] = loss_mask[num_examples:]

    return output
