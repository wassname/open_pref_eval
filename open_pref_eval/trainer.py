from dataclasses import dataclass, field
from datasets import Dataset, features
import tempfile
import torch
from trl import DPOConfig, DPOTrainer
from typing import Optional, Tuple, Dict, Any, List, Callable
import warnings
from einops import rearrange
from torch import Tensor
from jaxtyping import Float
from trl.trainer.dpo_trainer import DPODataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForCausalLM
from .helpers.hf_progbar import no_hf_tqdm


def get_supressed_activations(
    hs: Float[Tensor, "l b t h"], w_out, w_inv
) -> Float[Tensor, "l b t h"]:
    """
    Novel experiment: Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

    See the following references for more information:

    - https://arxiv.org/pdf/2401.12181
        - > Suppression neurons that are similar, except decrease the probability of a group of related tokens

    - https://arxiv.org/html/2406.19384
        - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4).

    - https://arxiv.org/pdf/2401.12181
        > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
    """
    with torch.no_grad():
        # here we pass the hs through the last layer, take a diff, and then project it back to find which activation changes lead to supressed
        hs2 = rearrange(hs[:, :, -1:], "l b t h -> (l b t) h")
        hs_out2 = torch.nn.functional.linear(hs2, w_out)
        hs_out = rearrange(
            hs_out2, "(l b t) h -> l b t h", l=hs.shape[0], b=hs.shape[1], t=1
        )
        diffs = hs_out[:, :, :].diff(dim=0)
        diffs2 = rearrange(diffs, "l b t h -> (l b t) h")
        # W_inv = get_cache_inv(w_out)

        diffs_inv2 = torch.nn.functional.linear(diffs2.to(dtype=w_inv.dtype), w_inv)
        diffs_inv = rearrange(
            diffs_inv2, "(l b t) h -> l b t h", l=hs.shape[0] - 1, b=hs.shape[1], t=1
        ).to(w_out.dtype)
        # TODO just return this?
        eps = 1.0e-1
        supressed_mask = (diffs_inv < -eps).to(hs.dtype)
        # supressed_mask = repeat(supressed_mask, 'l b 1 h -> l b t h', t=hs.shape[2])
    supressed_act = hs[1:] * supressed_mask
    return supressed_act

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
        tokenizer=tokenizer,
        train_dataset=dummy_dataset,
        eval_dataset=dummy_dataset,
    )
    return trainer



@dataclass
class OPEDataCollatorWithPadding(DPODataCollatorWithPadding):
    tokenizer: Optional[AutoTokenizer] = None
    tokenize_row: Optional[Callable] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenize
        tokenized_features = [self.tokenize_row(feature) for feature in features]
        return super().__call__(tokenized_features)


class OPETrainer(DPOTrainer):

    def __init__(self, *pargs, args: Optional[DPOConfig] = None, **kwargs):
        super().__init__(*pargs, args=args, **kwargs)

        # custom data collator that does tokenisation on the fly to save mem
        self.data_collator = OPEDataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
                tokenizer=self.tokenizer,
                tokenize_row=self.tokenize_row,
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


    def concatenated_forward(
        self, model, batch, model_kwargs = {}):
        """
        We modify this to simply return the logps and mask without reducing them
        """

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            # is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]        

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        per_token_logps, mask = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_t_logps = per_token_logps[:len_chosen]
        rejected_t_logps = per_token_logps[len_chosen:]

        chosen_mask = mask[:len_chosen]
        rejected_mask = mask[len_chosen:]
        all_logits = outputs.logits
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        all_hs = rearrange(list(outputs.hidden_states), "l b t h -> l b t h")

        Wo = model.get_output_embeddings().weight
        Wo_inv = torch.pinverse(Wo.clone().float()).detach()
        hs_sup = get_supressed_activations(all_hs, Wo, Wo_inv)
        hs_sup = rearrange(hs_sup, "l b t h -> b l t h")

        chosen_mask2 = mask[:len_chosen, :, None]
        rejected_mask2 = mask[len_chosen:, :, None]
        half_layer = hs_sup.shape[1] // 2
        chosen_hs = hs_sup[:len_chosen, half_layer:].sum(1) # sum over 2nd half of layers
        rejected_hs = hs_sup[len_chosen:, half_layer:].sum(1)
        # mean over tokens [b, t, h]
        chosen_hs = (chosen_hs[:, 1:] * chosen_mask2).sum(1) / chosen_mask2.sum(1)
        rejected_hs = (rejected_hs[:, 1:] * rejected_mask2).sum(1) / rejected_mask2.sum(1)
        # now should be [b, h]

        return (chosen_t_logps, rejected_t_logps, chosen_logits, rejected_logits, chosen_hs, rejected_hs, chosen_mask, rejected_mask)
