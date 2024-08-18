from datasets import load_dataset
from contextlib import contextmanager, nullcontext
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import Dataset
from trl import DPOTrainer
from typing import Optional, List, Union
from collections import OrderedDict
import numpy as np
from .datasets import get_default_datasets
from .trainer import get_dummy_trainer
from .helpers.peft import set_adapter


@torch.no_grad()
def eval_dpo_dataset(trainer: DPOTrainer, dataset: Union[Dataset,str]):
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

    assert trainer.loss_type == 'ipo', 'only ipo is supported, since it gives us the avg of logps, and is not biased by response length'
    
    compte_ref_context_manager = torch.cuda.amp.autocast if trainer._peft_has_been_casted_to_bf16 else nullcontext
    
    with compte_ref_context_manager():
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {dataset._info.dataset_name}")):

            # FIXME test
            # batch = trainer._prepare_inputs(batch)

            bs = batch['chosen_input_ids'].shape[0]
            i = bs * step + torch.arange(bs)


            def forward(trainer, model, batch):
                forward_output = trainer.concatenated_forward(model, batch)
                (
                    chosen_logps,
                    rejected_logps,
                ) = forward_output[:2]

                # Note: if we are using ipo or reprpo this will be adjusted for length, but otherwise not which would bias the results
                logratio = chosen_logps-rejected_logps

                # turn into list of dicts
                n = dict(
                    _logratio=logratio.detach().cpu().float().numpy(),
                    _chosen_logps=chosen_logps.detach().cpu().float().numpy(),
                    _l_chosen=(batch['chosen_labels']>0).sum(-1).detach().cpu().float().numpy(),
                    _l_rejected=(batch['rejected_labels']>0).sum(-1).detach().cpu().float().numpy(),
                    ds_i=i.numpy()
                )
                return [dict(
                    model=trainer.model.config._name_or_path,
                    # arrays
                    **{k:v[i] for k,v in n.items()}
                ) for i in range(bs)]
            
            if hasattr(model, 'peft_config') and hasattr(model, 'set_adapter'):
                # if model has peft adapters loop through them
                adapters = [None] +list(model.peft_config.keys())
                for adapter_name in adapters:
                    with set_adapter(model, adapter_name):
                        d = forward(trainer, model, batch)
                        for dd in d:
                            dd['adapter'] = adapter_name if adapter_name is not None else 'base'
                            data.append(dd)
            else:
                data += forward(trainer, model, batch)

    df = pd.DataFrame(data)
    df['correct'] = df['_logratio'] > 0.

    # prob mass on correct answer
    odds = np.exp(df['_logratio'])
    df['prob'] = odds / (1 + odds)

    df['dataset'] = dataset._info.dataset_name
    return df


def eval_dpo_datasets(datasets, trainer):

    dfs = []
    for dataset in datasets:
        df = eval_dpo_dataset(trainer, dataset)
        dfs.append(df)
    df = pd.concat(dfs)
    df['model'] = trainer.model.config._name_or_path
    return df

def evaluate_model(datasets: List[Dataset], trainer: Optional[DPOTrainer]=None, **kwargs):

    if trainer is None:
        trainer = get_dummy_trainer(**kwargs)

    df_raw = eval_dpo_datasets(datasets, trainer)

    # reorder df cols
    cols = ['model', 'dataset', 'ds_i', 'correct', 'prob']
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols+others]

    df_agg =  df_raw.groupby(['dataset'], dropna=False)[['correct', 'prob']].mean()
    df_agg['n'] = df_raw.groupby(['dataset'], dropna=False).size()
    # df_agg['model'] = trainer.model.config._name_or_path
    return df_agg, df_raw


def evaluate_models(datasets: List[Dataset], model_names: List[str], **kwargs):
    dfs = []
    for model_name in model_names:
        trainer = get_dummy_trainer(model_name=model_name, **kwargs)
        df_agg, df_raw = evaluate_model(datasets, trainer)
        dfs.append(df_agg)
    df_agg = pd.concat(dfs)
    return df_agg


def evaluate(model_names: List[str], datasets: Optional[List[Dataset]]=None, batch_size=4, **kwargs):
    """main class, rename args for clarity"""
    if datasets is None:
        datasets = get_default_datasets()
    return evaluate_models(per_device_eval_batch_size=batch_size, **kwargs)



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
