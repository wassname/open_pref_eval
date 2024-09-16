from datasets import load_dataset, Dataset
from contextlib import contextmanager, nullcontext
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from trl import DPOTrainer
from typing import Optional, List, Union, Callable
from collections import OrderedDict
import numpy as np
from jaxtyping import Float, Int

from .datasets import get_default_datasets, ds2name
from .trainer import get_dummy_trainer, OPETrainer
from .helpers.peft import set_adapter, is_peft_model, adapter_is_disabled
from .helpers.mem import clear_mem
from .scoring import score_1st_diverg, score_weighted, score_dpo, score_ipo

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



def extract_logps(trainer: OPETrainer, model: AutoModelForCausalLM, batch: dict, step: int, score_fn: Optional[Callable]=score_weighted
                  ) -> List[dict]:
    bs = batch['chosen_input_ids'].shape[0]
    i = bs * step + torch.arange(bs)
    forward_output = trainer.concatenated_forward(model, batch)
    (chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask) = forward_output

    # Here we decide how to reduce the per_token_logps to a single uncalibrated probability
    prob = score_fn(chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask)

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
def eval_dataset(trainer: OPETrainer, dataset: Union[Dataset,str], adapter_names:Optional[List[str]]= None, verbose=True, **kwargs) -> pd.DataFrame:
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

    if not verbose: tqdm = lambda x: x
    
    with compte_ref_context_manager():
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {ds2name(dataset)}")):
            batch = trainer._prepare_inputs(batch)

            if is_peft_model(model):
                # if model has peft adapters loop through them
                if adapter_names is None:
                    adapter_names = [None] +list(model.peft_config.keys())
                for adapter_name in adapter_names:
                    with set_adapter(trainer.model, adapter_name):
                        d = extract_logps(trainer, model, batch, step, **kwargs)
                        for dd in d:
                            dd['adapter'] = adapter_name if adapter_name is not None else 'base'
                            data.append(dd)
            else:
                data += extract_logps(trainer, trainer.model, batch, step, **kwargs)
    clear_mem()

    df = pd.DataFrame(data)
    df['correct'] = df['prob'] > 0.5

    df['dataset'] = ds2name(dataset)
    return df


def eval_datasets(datasets: List[Dataset], trainer: Optional[OPETrainer]=None, **kwargs) -> pd.DataFrame:
    dfs = []
    for dataset in datasets:
        df = eval_dataset(trainer, dataset, **kwargs)
        dfs.append(df)
    df = pd.concat(dfs)

    df['model'] = trainer.model.config._name_or_path # Error only has the base model
    return df

def evaluate_model(datasets: List[Dataset], trainer: Optional[OPETrainer]=None, model_kwargs={}, score_fn=score_weighted, **trainer_kwargs):
    trainer_kwargs = alias_trl_kwargs(trainer_kwargs)

    if trainer is None:
        trainer = get_dummy_trainer(model_kwargs=model_kwargs, **trainer_kwargs)

    df_raw = eval_datasets(datasets, trainer, score_fn=score_fn)

    # reorder df cols
    cols = ['model', 'dataset', 'ds_i', 'correct', 'prob']
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols+others]

    numeric_cols = list(set(['correct', 'prob']).intersection(df_raw.columns))
    agg_cols = ['dataset']
    if 'adapter' in df_raw.columns:
        agg_cols.append('adapter')

    df_agg =  df_raw.groupby(agg_cols, dropna=False)[numeric_cols].mean()
    df_agg['n'] = df_raw.groupby(agg_cols, dropna=False).size()
    clear_mem()
    return df_agg, df_raw


def evaluate_models(datasets: List[Dataset], model_names: List[str], **kwargs):
    dfs_raw = []
    for model_name in model_names:
        df_raw = evaluate_model(datasets=datasets, model_name=model_name, **kwargs)
        df_raw['model'] = model_name
        clear_mem()
        dfs_raw.append(df_raw)
    df_raw = pd.concat(dfs_raw)

    df_agg = df_raw.groupby(['model', 'dataset'], dropna=False)[['correct', 'prob']].mean()
    
    return df_agg, df_raw



def evaluate(model_names: List[str], datasets: Optional[List[Dataset]]=None, **kwargs):
    """main class, rename args for clarity"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names, datasets=datasets, **kwargs)

