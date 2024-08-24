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
from .trainer import get_dummy_trainer
from .helpers.peft import set_adapter, is_peft_model
from .helpers.calibrate import get_calibrator


def ds2name(dataset: Dataset) -> str:
    if dataset._info.splits is None:
        split = ''
    else:
        split=next(iter(dataset._info.splits.values())).name
    return f'{dataset.info.dataset_name} {dataset.info.config_name} {split}'

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

def calibrate_prob(df: pd.DataFrame, N:Union[bool,int]=False) -> pd.DataFrame:
    if N is False:
        N = 50
    
    df_train = df.iloc[:N]
    df_test = df.iloc[N:]
    calib = get_calibrator(df_train['prob'].values)
    df_test['prob_calib'] = calib.predict(df_test['prob'].values)
    
    # prevent data leakage
    df_train['prob_calib'] = np.nan
    df = pd.concat([df_train, df_test])
    return df

def extract_logps(trainer, model, batch, step):
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
def eval_dpo_dataset(trainer: DPOTrainer, dataset: Union[Dataset,str], calibrate: bool=False):
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

    if calibrate is not False:
        df = calibrate_prob(df, N=False)

    df['dataset'] = ds2name(dataset)
    return df


def eval_dpo_datasets(datasets, trainer, calibrate=False, **kwargs):

    dfs = []
    for dataset in datasets:
        df = eval_dpo_dataset(trainer, dataset, calibrate=False, **kwargs)
        dfs.append(df)
    df = pd.concat(dfs)

    # if we are doing multiple datasets, let override the calibration
    if calibrate:
        df = calibrate_prob(df, N=calibrate)

    df['model'] = trainer.model.config._name_or_path
    return df

def evaluate_model(datasets: List[Dataset], trainer: Optional[DPOTrainer]=None, calibrate=False, **kwargs):

    if trainer is None:
        trainer = get_dummy_trainer(**kwargs)

    df_raw = eval_dpo_datasets(datasets, trainer, calibrate=calibrate)

    # reorder df cols
    cols = ['model', 'dataset', 'ds_i', 'correct', 'prob']
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols+others]

    numeric_cols = list(set(['correct', 'prob', 'prob_calib']).intersection(df_raw.columns))

    df_agg =  df_raw.groupby(['dataset'], dropna=False)[numeric_cols].mean()
    df_agg['n'] = df_raw.groupby(['dataset'], dropna=False).size()
    # df_agg['model'] = trainer.model.config._name_or_path
    return df_agg, df_raw


def evaluate_models(datasets: List[Dataset], model_names: List[str], calibrate=False, **kwargs):
    dfs = []
    for model_name in model_names:
        trainer = get_dummy_trainer(model_name=model_name, **kwargs)
        df_agg, df_raw = evaluate_model(datasets, trainer, calibrate=calibrate)
        dfs.append(df_agg)
    df_agg = pd.concat(dfs)
    return df_agg


def evaluate(model_names: List[str], datasets: Optional[List[Dataset]]=None, batch_size=4, calibrate=False, **kwargs):
    """main class, rename args for clarity"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names,per_device_eval_batch_size=batch_size, datasets=datasets, calibrate=calibrate, **kwargs)



# def evaluate_adapters(model, tokenizer, datasets: Optional[List[Dataset]]=None, batch_size=4, **kwargs):
#     """
#     use the open_pref_eval library

#     to eval the model and it's adapters
#     """
#     1/0 # FIXME
#     if datasets is None:
#         datasets = get_default_datasets(N)

#     adapters = [None] +list(model.peft_config.keys())

#     dfs = []
#     for adapter in adapters:
#         print(f'Eval Adapter: {adapter}')
#         with set_adapter(model, adapter):
#             _, df_res2 = evaluate_model(datasets, model=model, tokenizer=tokenizer, per_device_eval_batch_size=batch_size, **kwargs)
#         df_res2['adapter'] = adapter if adapter is not None else 'base'
#         dfs.append(df_res2)
#     df_res = pd.concat(dfs)

#     df_agg =  df_res.groupby(['dataset', 'adapter'], dropna=False)['prob'].mean().unstack()
#     return df_agg, df_res
