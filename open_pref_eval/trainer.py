from dataclasses import dataclass, field
from datasets import Dataset
import tempfile
import torch
from trl import DPOConfig, DPOTrainer
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange
from torch import Tensor
from jaxtyping import Float
from transformers.data.data_collator import DataCollatorMixin
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from trl.trainer.utils import selective_log_softmax, flush_left
# from trl.trainer.dpo_trainer import DataCollatorForPreference
from .helpers.hf_progbar import no_hf_tqdm



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



def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

@dataclass
class DataCollatorForPreference(DataCollatorMixin):

    tokenizer: AutoTokenizer
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output


@dataclass
class OPEDataCollatorWithPadding(DataCollatorForPreference):

    tokenizer: AutoTokenizer
    max_prompt_length: Optional[int] = 512
    max_completion_length: Optional[int] = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenize each row, this does not add special tokens or mask
        tokenized_features = [tokenize_row(feature, self.tokenizer, max_prompt_length=self.max_prompt_length, max_completion_length=self.max_completion_length) for feature in features]

        # then collate normally, this adds mask
        return super().__call__(tokenized_features)
    

@staticmethod
def tokenize_row(features, processing_class, max_prompt_length, max_completion_length):
    """
    Tokenize a row of the dataset.

    Args:
        features (`dict[str, str]`):
            Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
        processing_class (`PreTrainedTokenizerBase`):
            Processing class used to process the data.
        max_prompt_length (`int` or `None`):
            Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
        max_completion_length (`int` or `None`):
            Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.

    Returns:
        `dict[str, list[int]]`:
            Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
            `"rejected_input_ids".
    """
    tokenizer = processing_class  # the processing class is a tokenizer
    prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    # Truncate prompt and completion sequences
    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    return {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def concatenated_inputs(
    batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
) -> dict[str, torch.LongTensor]:
    output = {}

    # For the prompt, the input_ids are the same for both the chosen and rejected responses
    output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )
    # Concatenate the chosen and rejected completions
    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    # FIXME pad_to_length is turning int to float
    output["completion_input_ids"] = torch.cat(
        (
            pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
        ),
    )
    output["completion_attention_mask"] = torch.cat(
        (
            pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
        ),
    )

    return output

def concatenated_forward(model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], padding_value: int = 0, device=None, dtype=None) -> dict[str, Float[Tensor, 'b t']]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.

    MODIFIED FROM TRL to return per token logps. I removed image, logits_to_keep, and paddning_free, truncation (should be done prior) logic to simplify it
    """
    num_examples = batch["prompt_input_ids"].shape[0]

    concatenated_batch = concatenated_inputs(batch, padding_value=padding_value)

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
