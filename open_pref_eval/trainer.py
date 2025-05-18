from dataclasses import dataclass, field
from datasets import Dataset, features
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
from functools import partial
from trl.trainer.dpo_trainer import DataCollatorForPreference
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.utils import selective_log_softmax, flush_left
from .helpers.hf_progbar import no_hf_tqdm


def alias_trl_kwargs(kwargs):
    """We take in transformers and trl trainer args, which are obscure, so we offer aliases"""
    popping = {
        # alias: full_kargs
        'batch_size': 'per_device_eval_batch_size',
    }
    aliasing = {
        'bf16': 'bf16_full_eval',
        'fp16': 'fp16_full_eval',
    }
    for k,v in popping.items():
        if k in kwargs:
            if not v in kwargs:
                kwargs[v] = kwargs.pop(k)
    for k,v in aliasing.items():
        if k in kwargs:
            if not v in kwargs:
                kwargs[v] = kwargs[k]
    return kwargs

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
class OPEConfig(DPOConfig):
    loss_type: str = 'ipo'
    max_length: int = 512
    max_prompt_length: int = 128
    disable_tqdm=True
    should_save=False
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    dataloader_pin_memory: bool = True

    use_weighting: bool = True


@no_hf_tqdm()
def get_dummy_trainer(model=None, tokenizer=None, model_name:Optional[str]=None, per_device_eval_batch_size=8, model_kwargs={}, **kwargs):
    """
    Make a dummy trainer, 

    For keyword arguments, see 
    - [transformers.TrainingArguments](https://huggingface.co/docs/transformers/v4.43.3/en/main_classes/trainer#transformers.TrainingArguments)
    - [trl.DPOConfig](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig)

    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = OPEConfig(
            output_dir=tmp_dir,
            per_device_eval_batch_size=per_device_eval_batch_size,
            **kwargs
        )

    if model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    **model_kwargs,
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        raise ValueError('model or model_name must be provided')

    # we use a TRL class
    trainer = OPETrainer(
        model=model,
        ref_model=None,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dummy_dataset,
        eval_dataset=dummy_dataset,
    )
    return trainer



@dataclass
class OPEDataCollatorWithPadding(DataCollatorForPreference):
    tokenize_row: Optional[Callable] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenize each row, this does not add special tokens or mask
        tokenized_features = [self.tokenize_row(feature) for feature in features]
        # then collate normally, this adds mask
        return super().__call__(tokenized_features)


class OPETrainer(DPOTrainer):

    def __init__(self, *pargs, args: Optional[DPOConfig] = None, **kwargs):
        super().__init__(*pargs, args=args, **kwargs)

        # custom data collator that does tokenisation on the fly to save mem
        # see usage https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L554
        self.data_collator = OPEDataCollatorWithPadding(
                pad_token_id=self.processing_class.pad_token_id,
                # label_pad_token_id=args.label_pad_token_id,
                # is_encoder_decoder=self.is_encoder_decoder,
                # tokenizer=self.processing_class,
                tokenize_row=partial(self.tokenize_row, processing_class=self.processing_class, max_prompt_length=args.max_prompt_length, max_completion_length=args.max_length - args.max_prompt_length, add_special_tokens=False),
            )

        if args.remove_unused_columns:
            args.remove_unused_columns = False
            # warn users
            warnings.warn(
                "When using OPEDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                " we have set it for you, but you should do it yourself in the future.",
                UserWarning,
            )

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        We modify this to return the logps and mask without reducing them
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps, loss_mask
    
    # @staticmethod
    # def tokenize_row(batch, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
    #     """for some reason TRL changed the class to not return the mask... I should move away from TRL it's too complex."""

    #     tokenizer = processing_class
    #     chosen_tokens = tokenizer(
    #         features["chosen"], truncation=True, max_length=max_completion_length, add_special_tokens=False, truncate_side="right"
    #     )
    #     rejected_tokens = tokenizer(
    #         features["rejected"], truncation=True, max_length=max_completion_length, add_special_tokens=False, truncate_side="right"
    #     )
    #     prompt_tokens = tokenizer(
    #         features["prompt"], truncation=True, max_length=max_prompt_length, add_special_tokens=False, truncate_side="left"
    #     )

    #     batch["chosen_input_ids"] = chosen_tokens["input_ids"]
    #     batch["rejected_input_ids"] = rejected_tokens["input_ids"]
    #     batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    #     batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
    #     batch["chosen_attention_mask"] = chosen_tokens["attention_mask"]
    #     batch["rejected_attention_mask"] = rejected_tokens["attention_mask"]
    #     return batch

    # def concatenated_forward(
    #     self, model, batch, model_kwargs = {}):
    #     """
    #     We modify this to simply return the logps and mask without reducing them
    #     """

    #     concatenated_batch = self.concatenated_inputs(
    #         batch,
    #         padding_value=self.padding_value,
    #     )
    #     len_chosen = batch["concatenated_input_ids"].shape[0]        

    #     if self.is_encoder_decoder:
    #         model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
    #         model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

    #     outputs = model(
    #         concatenated_batch["concatenated_input_ids"],
    #         attention_mask=concatenated_batch["concatenated_attention_mask"],
    #         use_cache=False,
    #         **model_kwargs,
    #     )
    #     all_logits = outputs.logits

    #     per_token_logps, mask = self.get_batch_logps(
    #         all_logits,
    #         concatenated_batch["concatenated_labels"],
    #         # average_log_prob=self.loss_type == "ipo",
    #         is_encoder_decoder=self.is_encoder_decoder,
    #         label_pad_token_id=self.label_pad_token_id,
    #     )

    #     chosen_t_logps = per_token_logps[:len_chosen]
    #     rejected_t_logps = per_token_logps[len_chosen:]

    #     chosen_mask = mask[:len_chosen]
    #     rejected_mask = mask[len_chosen:]

    #     return (chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask)
    

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

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

        # Truncate right
        if self.max_length is not None:
            if self.truncation_mode == "keep_end":
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                loss_mask = loss_mask[:, -self.max_length :]
            elif self.truncation_mode == "keep_start":
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                    "'keep_start']."
                )

        if self.use_logits_to_keep:
            # Compute logits_to_keep based on loss_mask pattern:
            # [[0, 0, 0, x, x, x, x],
            #  [0, 0, 0, x, x, x, 0]]
            #         ^ start computing logits from here ([:, -(7-3+1):])
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
            model_kwargs["logits_to_keep"] = logits_to_keep

        if self.padding_free:
            # Flatten the input_ids, position_ids, and loss_mask
            # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
            #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        # Offset the logits by one to align with the labels
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            # Align labels with logits
            # logits:    -,  -, [x2, x3, x4, x5, x6]
            #                     ^ --------- ^       after logits[:, :-1, :]
            # labels:   [y0, y1, y2, y3, y4, y5, y6]
            #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
            # loss_mask: [0,  0,  0,  1,  1,  1,  1]
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = per_token_logps[:num_examples]
        output["rejected_logps"] = per_token_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits
        output["chosen_mask"] = loss_mask[:num_examples]
        output["rejected_mask"] = loss_mask[num_examples:]

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
