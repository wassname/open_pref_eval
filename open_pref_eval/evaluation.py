from datasets import load_dataset, Dataset
from contextlib import contextmanager, nullcontext
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import Dataset
from trl import DPOTrainer
from typing import Optional, List, Union
from collections import OrderedDict
import numpy as np
from jaxtyping import Float, Int

from .datasets import get_default_datasets
from .trainer import get_dummy_trainer, OPETrainer
from .helpers.peft import set_adapter, is_peft_model
from .helpers.mem import clear_mem

def alias_trl_kwargs(kwargs):
    """We take in transformers and trl trainer args, which are obscure, so we offer aliases"""
    mapping = {
        # alias: full_kargs
        'batch_size': 'per_device_eval_batch_size',
        'bf16': 'bf16_full_eval',
        'fp16': 'fp16_full_eval',
    }
    for k,v in mapping.items():
        if k in kwargs:
            kwargs[v] = kwargs.pop(k)
    return kwargs

def ds2name(dataset: Dataset) -> str:
    if dataset._info.splits is None:
        split = ''
    else:
        split=next(iter(dataset._info.splits.values())).name

    config_name = dataset.info.config_name
    if config_name == 'default':
        config_name = ''
    
    return f'{dataset.info.dataset_name}-{config_name}-{split}'

def first_nonzero(x: Float[Tensor, 'b t'], dim=1) -> Float[Tensor, 'b']:
    """get the first non zero element in a tensor"""
    return x[torch.arange(x.shape[0]), (x != 0).float().argmax(dim=dim)]

def score_1st_diverg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    """
    calculate if the chosen completion is higher than the rejected, using first divering token

    return uncalibrated probability
    """
    m = mask_c * mask_r
    logratio = (logp_c - logp_r) * m
    return torch.sigmoid(first_nonzero(logratio))

# def score_dpo(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
#     """
#     calculate if the chosen completion is higher than the rejected, using DPO

#     return uncalibrated probability
#     """
#     # get the total logprob of the completion
#     c = (logp_c * mask_c).sum(-1)
#     r = (logp_r * mask_r).sum(-1)

#     # and the ratio in logspace
#     logratio = c - r

#     # return uncalibrated probability
#     return torch.sigmoid(logratio.sum(1))


# def score_ipo(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
#     # get the avg logprob of the completion
#     c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1)
#     r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1)

#     # and the ratio in logspace
#     logratio = c - r

#     # return uncalibrated probability
#     return torch.sigmoid(logratio.sum(1))


# def score_cumsum(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
#     # get the avg logprob over the cumulative logprob of each token, this means the initial tokens are weighted higher, but all tokens have an influence
#     c = (logp_c * mask_c).cumsum(-1).sum(-1) / mask_c.sum(-1)
#     r = (logp_r * mask_r).cumsum(-1).sum(-1) / mask_r.sum(-1)

#     # and the ratio in logspace
#     logratio = c - r

#     # return uncalibrated probability
#     return torch.sigmoid(logratio.sum(1))

def extract_logps(trainer: OPETrainer, model: AutoModelForCausalLM, batch: dict, step: int) -> List[dict]:
    bs = batch['chosen_input_ids'].shape[0]
    i = bs * step + torch.arange(bs)
    forward_output = trainer.concatenated_forward(model, batch)
    (chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask) = forward_output

    # Here we decide how to reduce the per_token_logps to a single uncalibrated probability
    prob = score_1st_diverg(chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask)

    # logprob of whole completion
    chosen_logp = (chosen_t_logps * chosen_mask).sum(1)
    rejected_logp = (rejected_t_logps * rejected_mask).sum(1)

    # turn into list of dicts
    n = dict(
        prob=prob.detach().cpu().float().numpy(),

        # debug: logprobs of completions, can be used to check if coherency is maintained
        _chosen_logps=chosen_logp.detach().cpu().float().numpy(),
        _rejected_logps=rejected_logp.detach().cpu().float().numpy(),

        # debug: completion length, for checking if the model is biased
        _l_chosen=(batch['chosen_labels']>0).sum(-1).detach().cpu().float().numpy(),
        _l_rejected=(batch['rejected_labels']>0).sum(-1).detach().cpu().float().numpy(),

        # metadata
        ds_i=i.numpy(),
    )
    return [dict(
        model=trainer.model.config._name_or_path,
        # arrays
        **{k:v[i] for k,v in n.items()}
    ) for i in range(bs)]

@torch.no_grad()
def eval_dataset(trainer: OPETrainer, dataset: Union[Dataset,str]) -> pd.DataFrame:
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    if isinstance(dataset, str):
        dataset_name, split = dataset.split('#')
        dataset = load_dataset(dataset_name, split=split, keep_in_memory=False)

    model = trainer.model
    model.eval()
    model.config.use_cache = False


    data = []
    # use hf dpo trainer to tokenizer, and make loader
    dataset2 = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=10)
    eval_dataloader = trainer.get_eval_dataloader(dataset2)
    
    compte_ref_context_manager = torch.cuda.amp.autocast if trainer._peft_has_been_casted_to_bf16 else nullcontext
    
    with compte_ref_context_manager():
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {ds2name(dataset)}")):
            batch = trainer._prepare_inputs(batch)

            if is_peft_model(model):
                # if model has peft adapters loop through them
                adapters = [None] +list(model.peft_config.keys())
                for adapter_name in adapters:
                    with set_adapter(model, adapter_name):
                        d = extract_logps(trainer, model, batch, step)
                        for dd in d:
                            dd['adapter'] = adapter_name if adapter_name is not None else 'base'
                            data.append(dd)
            else:
                data += extract_logps(trainer, model, batch, step)

    df = pd.DataFrame(data)
    df['correct'] = df['prob'] > 0.5

    df['dataset'] = ds2name(dataset)
    return df


def eval_datasets(datasets: List[Dataset], trainer: Optional[OPETrainer]=None) -> pd.DataFrame:
    dfs = []
    for dataset in datasets:
        df = eval_dataset(trainer, dataset)
        dfs.append(df)
    df = pd.concat(dfs)

    df['model'] = trainer.model.config._name_or_path # Error only has the base model
    return df

def evaluate_model(datasets: List[Dataset], trainer: Optional[OPETrainer]=None, model_kwargs={}, **trainer_kwargs):
    trainer_kwargs = alias_trl_kwargs(trainer_kwargs)

    if trainer is None:
        trainer = get_dummy_trainer(model_kwargs=model_kwargs, **trainer_kwargs)

    df_raw = eval_datasets(datasets, trainer)

    # reorder df cols
    cols = ['model', 'dataset', 'ds_i', 'correct', 'prob']
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols+others]

    numeric_cols = list(set(['correct', 'prob']).intersection(df_raw.columns))

    df_agg =  df_raw.groupby(['dataset'], dropna=False)[numeric_cols].mean()
    df_agg['n'] = df_raw.groupby(['dataset'], dropna=False).size()
    return df_agg, df_raw


def evaluate_models(datasets: List[Dataset], model_names: List[str], **kwargs):
    dfs = []
    dfs_raw = []
    for model_name in tqdm(model_names, unit='model'):
        df_agg, df_raw = evaluate_model(datasets, model_name=model_name, **kwargs)
        df_agg['model'] = model_name
        df_raw['model'] = model_name
        clear_mem()
        dfs.append(df_agg)
        dfs_raw.append(df_raw)
    df_agg = pd.concat(dfs)
    df_raw = pd.concat(dfs_raw)
    
    return df_agg, df_raw



def evaluate(model_names: List[str], datasets: Optional[List[Dataset]]=None, batch_size=4, **kwargs):
    """main class, rename args for clarity"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names,per_device_eval_batch_size=batch_size, datasets=datasets, **kwargs)

