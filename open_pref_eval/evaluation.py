from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import Dataset
from trl import DPOTrainer
from typing import Optional, List
from collections import OrderedDict
import numpy as np
from .trainer import get_dummy_trainer


@torch.no_grad()
def eval_dpo_dataset(trainer: DPOTrainer, dataset: Dataset):
    """
    We eval the prob_chosen/prob_rejected for each sample in the dataset (per token)

    Must have cols chosen and rejected just like the trl dpotrainer

    see trainer.evaluation_loop
    """
    model = trainer.model
    model.eval()
    model.config.use_cache = False


    data = []
    # use hf dpo trainer to tokenizer, and make loader
    dataset = dataset.map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc, writer_batch_size=10)
    # eval_dataloader = trainer.get_eval_dataloader(dataset)
    eval_dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=trainer.args.eval_batch_size, collate_fn=trainer.data_collator)

    # model = trainer._wrap_model(trainer.model, training=False, dataloader=eval_dataloader)
    # model = trainer.accelerator.prepare_model(model, evaluation_mode=True)

    assert trainer.loss_type == 'ipo', 'only ipo is supported, since it gives us the avg of logps, and is not biased by response length'
    
    with torch.cuda.amp.autocast():
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

            batch['chosen_input_ids'].shape
            batch['rejected_input_ids'].shape
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
    df['correct'] = df['logratio'] > 0.

    # prob mass on correct answer
    odds = np.exp(df['logratio'])
    df['prob'] = odds / (1 + odds)

    # reorder df cols
    cols = ['dataset', 'i', 'correct', 'prob']
    others = [c for c in df.columns if c not in cols]
    df = df[cols+others]
    return df


def eval_dpo_datasets(datasets, trainer):

    dfs = []
    for dataset in datasets:
        df = eval_dpo_dataset(trainer, dataset)
        dfs.append(df)
        df['dataset'] = dataset._info.dataset_name
        dfs.append(df)
    df = pd.concat(dfs)

    return df

def evaluate(model: Optional[AutoModelForCausalLM], tokenizer: Optional[AutoTokenizer], datasets: List[Dataset], trainer: Optional[DPOTrainer]=None):

    if trainer is None:
        if model is None or tokenizer is None:
            raise ValueError('must provide model and tokenizer or trainer')
        trainer = get_dummy_trainer(model, tokenizer)

    df_raw = eval_dpo_datasets(datasets, trainer)

    df_agg =  df_raw.groupby(['dataset'], dropna=False).mean()[['correct', 'prob']]
    return df_agg, df_raw

