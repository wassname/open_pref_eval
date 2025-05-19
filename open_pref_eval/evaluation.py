import itertools
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .datasets import Dataset, ds2name, get_default_datasets
from .helpers.mem import clear_mem
from .helpers.peft import is_peft_model, set_adapter
from .scoring import score_weighted
from .trainer import DataCollatorForPreference, concatenated_forward


def extract_logps(
    model, batch, step: int, score_fn: Callable = score_weighted, include_raw=False
):
    bs = batch["chosen_input_ids"].shape[0]
    i = bs * step + torch.arange(bs)
    model.eval()
    # device_type = "xpu" if is_torch_xpu_available() else "cuda"
    # dtype = torch.bfloat16 if trainer._peft_has_been_casted_to_bf16 else torch.float32
    # compte_ref_context_manager = torch.amp.autocast(device=device_type, dtype=dtype) if trainer._peft_has_been_casted_to_bf16 else nullcontext()
    # with torch.no_grad(), compte_ref_context_manager:
    forward_output = concatenated_forward(model, batch)

    chosen_t_logps = forward_output["chosen_logps"]
    rejected_t_logps = forward_output["rejected_logps"]
    chosen_mask = forward_output["chosen_mask"]
    rejected_mask = forward_output["rejected_mask"]

    # Here we decide how to reduce the per_token_logps to a single uncalibrated probability
    prob = score_fn(chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask)
    assert torch.isfinite(prob).all(), f"Prob is not finite: {prob}"

    # logprob of whole completion
    chosen_logp = (chosen_t_logps * chosen_mask).sum(1)
    rejected_logp = (rejected_t_logps * rejected_mask).sum(1)

    # calculate perplexity
    chosen_ppl = torch.exp(-chosen_logp / chosen_mask.sum(1))
    rejected_ppl = torch.exp(-rejected_logp / rejected_mask.sum(1))

    # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
    # per_token_logps = torch.concatenate([chosen_t_logps, rejected_t_logps], dim=0)
    adj_cho = torch.logsumexp(
        2 * chosen_t_logps, dim=-1, keepdim=True
    )  # same as sum(probs**2) in log space
    chosen_weights = ((chosen_t_logps - adj_cho) * chosen_mask).sum(
        -1
    ) / chosen_mask.sum(-1)

    adj_rej = torch.logsumexp(
        2 * rejected_t_logps, dim=-1, keepdim=True
    )  # same as sum(probs**2) in log space
    rejected_weights = ((rejected_t_logps - adj_rej) * rejected_mask).sum(
        -1
    ) / rejected_mask.sum(-1)
    policy_weights = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

    # turn into list of dicts
    n = dict(
        prob=prob,
        # debug: logprobs and ppl of completions, can be used to check if coherency is maintained
        _chosen_logps=chosen_logp,
        _rejected_logps=rejected_logp,
        _chosen_ppl=chosen_ppl,
        _rejected_ppl=rejected_ppl,
        # debug: completion length, for checking if the model is biased
        _l_chosen=(batch["chosen_input_ids"] > 0).sum(-1),
        _l_rejected=(batch["rejected_input_ids"] > 0).sum(-1),
        _policy_weights=policy_weights,
    )
    if include_raw:
        n.update(
            __chosen_logps=chosen_t_logps,
            __rejected_logps=rejected_t_logps,
            __chosen_logits=forward_output["mean_chosen_logits"],
            __rejected_logits=forward_output["mean_rejected_logits"],
            __chosen_mask=chosen_mask,
            __rejected_mask=rejected_mask,
        )
        clear_mem()
    n = {k: v.detach().cpu().numpy() for k, v in n.items()}
    # metadata
    n["ds_i"] = i.cpu().numpy()
    return n


@torch.no_grad()
def eval_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Union[Dataset, str],
    adapter_names: Optional[List[str]] = None,
    max_prompt_length: int = 512,
    max_length: Optional[int] = 1024,
    batch_size=2,
    verbose=1,
    **kwargs,
) -> pd.DataFrame:
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    if isinstance(dataset, str):
        dataset_name, split = dataset.split("#")
        dataset: Dataset = load_dataset(dataset_name, split=split, keep_in_memory=False)

    model.eval()
    model.config.use_cache = False
    dsname = ds2name(dataset)

    data = []

    # Note: normally tl.dpo requires pretokenisation, but that would cause memory problems with large datasets so it's been moved to the collator
    # with no_hf_tqdm():
    # dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=100, desc='tokenize', keep_in_memory=False)

    assert max_length > max_prompt_length, (
        f"max_length {max_length} must be greater than max_prompt_length {max_prompt_length}"
    )
    data_collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_length - max_prompt_length,
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    for step, batch in enumerate(
        tqdm(eval_dataloader, desc=f"Eval {ds2name(dataset)}", disable=verbose < 2)
    ):
        if is_peft_model(model):
            # if model has peft adapters loop through them
            if adapter_names is None:
                adapter_names: list = [None] + list(model.peft_config.keys())
            for adapter_name in adapter_names:
                with set_adapter(model, adapter_name):
                    d = extract_logps(model, batch, step, **kwargs)
                    adapter_name if adapter_name is not None else "base"
                    d["adapter"] = [adapter_name] * len(d["prob"])
                    data.append(d)
        else:
            d = extract_logps(model, batch, step, **kwargs)
            data.append(d)

    # so here we have a list of dict of lists. We want concat each key to get a dict of lists
    keys = data[0].keys()
    data2 = {k: itertools.chain(*[d[k] for d in data]) for k in keys}
    # data2 = {k: torch.concat([d[k] for d in data], 0).cpu().float().numpy() for k in list(keys)[:1]}
    # data2['adapter'] = list(itertools.chain(*[d['adapter'] for d in data]))
    # df = Dataset.from_dict(data2).with_format('pandas')
    df = pd.DataFrame(data2)
    # df = pd.concat(data)

    # TODO I'd like a robust way to calibrate logprobs so we can get a better signal with a much smaller dataset
    df["correct"] = df["prob"] > 0.5
    df["model"] = model.config._name_or_path
    df["dataset"] = dsname

    return df


def eval_datasets(
    model, tokenizer, datasets: List[Dataset], verbose=1, **kwargs
) -> pd.DataFrame:
    dfs = []
    for dataset in tqdm(datasets, disable=not verbose, unit="dataset"):
        df = eval_dataset(model, tokenizer, dataset, verbose=verbose, **kwargs)
        dfs.append(df)
        clear_mem()
    df = pd.concat(dfs)

    df["model"] = model.config._name_or_path  # Error only has the base model
    return df


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    datasets: List[Dataset],
    score_fn=score_weighted,
    verbose=1,
    **kwargs,
) -> pd.DataFrame:
    df_raw = eval_datasets(
        model, tokenizer, datasets, score_fn=score_fn, verbose=verbose, **kwargs
    )

    # reorder df cols
    cols = ["model", "dataset", "ds_i", "correct", "prob"]
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols + others]

    numeric_cols = list(set(["correct", "prob"]).intersection(df_raw.columns))
    agg_cols = ["dataset"]
    if "adapter" in df_raw.columns:
        agg_cols.append("adapter")

    df_agg = df_raw.groupby(agg_cols, dropna=False)[numeric_cols].mean()
    df_agg["n"] = df_raw.groupby(agg_cols, dropna=False).size()
    clear_mem()
    return df_agg, df_raw


def evaluate_models(
    datasets: List[Dataset], model_names: List[str], model_kwargs={}, **kwargs
):
    dfs_raw = []
    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        df_raw, _ = evaluate_model(model, tokenizer, datasets=datasets, **kwargs)
        df_raw["model"] = model_name
        clear_mem()
        dfs_raw.append(df_raw)
    df_raw = concatenate_datasets(dfs_raw)

    df_raw = df_raw.select_columns(
        ["model", "dataset", "ds_i", "correct", "prob"]
    ).to_pandas()
    # .to_pandas()

    df_agg = df_raw.groupby(["model", "dataset"], dropna=False)[
        ["correct", "prob"]
    ].mean()

    return df_agg, df_raw


def evaluate(
    model_names: List[str], datasets: Optional[List[Dataset]] = None, **kwargs
):
    """main class"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names, datasets=datasets, **kwargs)
