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
from .trainer import get_dummy_trainer


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
        for step, batch in enumerate(eval_dataloader):
            # batch = trainer._prepare_inputs(batch)

            forward_output = trainer.concatenated_forward(model, batch)
            (
                chosen_logps,
                rejected_logps,
            ) = forward_output[:2]
            # chosen_logavgp, rejected_logavgp = forward_output[4:6]

            # Note: if we are using ipo or reprpo this will be adjusted for length, but otherwise not which would bias the results
            logratio = chosen_logps-rejected_logps

            bs = batch['chosen_input_ids'].shape[0]
            i = bs * step + torch.arange(bs)
            data.append(dict(
                _logratio=logratio.detach().cpu().float(),
                _chosen_logps=chosen_logps.detach().cpu().float(),
                _l_chosen=(batch['chosen_labels']>0).sum(-1).detach().cpu().float(),
                _l_rejected=(batch['rejected_labels']>0).sum(-1).detach().cpu().float(),
                ds_i=i
            ))
    # now concat the elements of data
    data = {k:torch.cat([d[k] for d in data], dim=0).numpy() for k in data[0].keys()}

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

def evaluate(datasets: List[Dataset], trainer: Optional[DPOTrainer]=None, **kwargs):

    if trainer is None:
        trainer = get_dummy_trainer(**kwargs)

    df_raw = eval_dpo_datasets(datasets, trainer)

    # reorder df cols
    cols = ['model', 'dataset', 'ds_i', 'correct', 'prob']
    others = [c for c in df_raw.columns if c not in cols]
    df_raw = df_raw[cols+others]

    df_agg =  df_raw.groupby(['dataset'], dropna=False)[['correct', 'prob']].mean()
    df_agg['n'] = df_raw.groupby(['dataset'], dropna=False).size()
    df_agg['model'] = trainer.model.config._name_or_path
    return df_agg, df_raw

# TODO eval over all model adapters
# TODO default datasets
