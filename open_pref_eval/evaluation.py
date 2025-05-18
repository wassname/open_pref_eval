from datasets import load_dataset, Dataset, concatenate_datasets
from contextlib import contextmanager, nullcontext
import pandas as pd
import torch
from transformers.utils import is_peft_available, is_torch_xpu_available
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tqdm.auto import tqdm
from trl import DPOTrainer
from typing import Optional, List, Union, Callable
from collections import OrderedDict
import numpy as np
from jaxtyping import Float, Int
import itertools
from einops import rearrange, reduce, repeat

from .datasets import get_default_datasets, ds2name, Dataset
from .trainer import get_dummy_trainer, OPETrainer, alias_trl_kwargs
from .helpers.peft import set_adapter, is_peft_model, adapter_is_disabled
from .helpers.mem import clear_mem
from .scoring import score_1st_diverg, score_weighted, score_preferences, score_ipo
from .helpers.hf_progbar import no_hf_tqdm
from .helpers.calibrate import get_calibrator



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
    
    df_train = df.iloc[:N].copy()
    df_test = df.iloc[N:].copy()
    calib = get_calibrator(df_train['prob'].values)
    if hasattr(calib, 'predict_proba'):
        df_test['prob_calib'] = calib.predict_proba(df_test['prob'].values)[:,1]
    else:
        df_test['prob_calib'] = calib.predict(df_test['prob'].values)
    
    # prevent data leakage
    df_train['prob_calib'] = np.nan
    df = pd.concat([df_train, df_test])
    return df


def extract_logps(trainer, model, batch, step, score_fn: Callable=score_weighted, include_raw=False):
    bs = batch['chosen_input_ids'].shape[0]
    i = bs * step + torch.arange(bs)
    model.eval()
    device_type = "xpu" if is_torch_xpu_available() else "cuda"
    compte_ref_context_manager = torch.amp.autocast(device_type) if trainer._peft_has_been_casted_to_bf16 else nullcontext()
    with torch.no_grad(), compte_ref_context_manager:
        forward_output = trainer.concatenated_forward(model, batch)

    chosen_t_logps= forward_output["chosen_logps"]
    rejected_t_logps= forward_output["rejected_logps"]
    chosen_logits= forward_output["mean_chosen_logits"]
    rejected_logits= forward_output["mean_rejected_logits"]
    chosen_mask = forward_output["chosen_mask"]
    rejected_mask = forward_output["rejected_mask"]

    # Here we decide how to reduce the per_token_logps to a single uncalibrated probability
    prob = score_fn(chosen_t_logps, rejected_t_logps, chosen_mask, rejected_mask)

    # logprob of whole completion
    chosen_logp = (chosen_t_logps * chosen_mask).sum(1)
    rejected_logp = (rejected_t_logps * rejected_mask).sum(1)

    # calculate perplexity
    chosen_ppl = torch.exp(-chosen_logp / chosen_mask.sum(1))
    rejected_ppl = torch.exp(-rejected_logp / rejected_mask.sum(1))


    # turn into list of dicts
    n = dict(
        prob=prob,

        # debug: logprobs and ppl of completions, can be used to check if coherency is maintained
        _chosen_logps=chosen_logp,
        _rejected_logps=rejected_logp,

        _chosen_ppl=chosen_ppl,
        _rejected_ppl=rejected_ppl,

        # debug: completion length, for checking if the model is biased
        _l_chosen=(batch['chosen_input_ids']>0).sum(-1),
        _l_rejected=(batch['rejected_input_ids']>0).sum(-1),


    )
    if include_raw:
        n.update(
            __chosen_logps=chosen_t_logps,
            __rejected_logps=rejected_t_logps,
            __chosen_logits=chosen_logits,
            __rejected_logits=rejected_logits,
            # __chosen_hs=chosen_hs,
            # __rejected_hs=rejected_hs,
            __chosen_mask=chosen_mask,
            __rejected_mask=rejected_mask
        )
        clear_mem()
    n = {k:v.detach().cpu() for k,v in n.items()}
    # metadata
    n['ds_i'] = i.cpu()
    return n

@torch.no_grad()
def eval_dataset(trainer: OPETrainer, dataset: Union[Dataset,str], adapter_names:Optional[List[str]]= None, verbose=1, calibrate=False, **kwargs) -> pd.DataFrame:
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    if isinstance(dataset, str):
        dataset_name, split = dataset.split('#')
        dataset: Dataset = load_dataset(dataset_name, split=split, keep_in_memory=False)

    model: PreTrainedModel = trainer.model
    model.eval()
    model.config.use_cache = False
    dsname = ds2name(dataset)

    data = []

    # Note: normally tl.dpo requires pretokenisation, but that would cause memory problems with large datasets so it's been moved to the collator
    # with no_hf_tqdm():
        # dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=100, desc='tokenize', keep_in_memory=False)
    
    eval_dataloader = trainer.get_eval_dataloader(dataset)

    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {ds2name(dataset)}", disable=verbose<2)):
        batch = trainer._prepare_inputs(batch)

        if is_peft_model(model):
            # if model has peft adapters loop through them
            if adapter_names is None:
                adapter_names: list = [None] +list(model.peft_config.keys())
            for adapter_name in adapter_names:
                with set_adapter(trainer.model, adapter_name):
                    d = extract_logps(trainer, model, batch, step, **kwargs)
                    d['adapter'] = adapter_name if adapter_name is not None else 'base'
                    data.append(d)
        else:
            d = extract_logps(trainer, model, batch, step, **kwargs)
            data.append(d)

    # so here we have a list of dict of lists. We want concat each key to get a dict of lists
    keys = data[0].keys()
    data = {k: itertools.chain(*[d[k] for d in data]) for k in keys}
    df = Dataset.from_dict(data)
    # df = pd.concat(data)

    # TODO I'd like a robust way to calibrate logprobs so we can get a better signal with a much smaller dataset
    df['correct'] = df['prob'] > 0.5
    df['model'] = trainer.model.config._name_or_path
    df['dataset'] = dsname


    if calibrate is not False:
        df = calibrate_prob(df, N=False)
    return df


def eval_datasets(datasets: List[Dataset], trainer: OPETrainer, verbose=1, calibrate=False, **kwargs) -> pd.DataFrame:
    dfs = []
    for dataset in tqdm(datasets, disable=not verbose, unit='dataset'):
        df = eval_dataset(trainer, dataset, verbose=verbose, calibrate=False, **kwargs)
        dfs.append(df)
        clear_mem()
    df = pd.concat(dfs)

    # if we are doing multiple datasets, let override the calibration
    if calibrate:
        df = calibrate_prob(df, N=calibrate)

    df['model'] = trainer.model.config._name_or_path # Error only has the base model
    return df

def evaluate_model(datasets: List[Dataset], trainer: Optional[OPETrainer]=None, model_kwargs={}, score_fn=score_weighted, verbose=1, calibrate=False, eval_kwargs={}, **trainer_kwargs, ) -> pd.DataFrame:
    trainer_kwargs = alias_trl_kwargs(trainer_kwargs)

    if trainer is None:
        with no_hf_tqdm():
            trainer = get_dummy_trainer(model_kwargs=model_kwargs, **trainer_kwargs)

    df_raw = eval_datasets(datasets, trainer, score_fn=score_fn, verbose=verbose, calibrate=calibrate, **eval_kwargs)

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
        df_raw, _ = evaluate_model(datasets=datasets, model_name=model_name, **kwargs)
        df_raw['model'] = model_name
        clear_mem()
        dfs_raw.append(df_raw)
    df_raw = concatenate_datasets(dfs_raw)
    
    df_raw = df_raw.select_columns(['model', 'dataset', 'ds_i', 'correct', 'prob']).to_pandas()
    #.to_pandas()

    df_agg = df_raw.groupby(['model', 'dataset'], dropna=False)[['correct', 'prob']].mean()

    return df_agg, df_raw


def evaluate(model_names: List[str], datasets: Optional[List[Dataset]]=None, **kwargs):
    """main class"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(model_names=model_names, datasets=datasets, **kwargs)

