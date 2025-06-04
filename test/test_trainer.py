import tempfile
import unittest

import pytest
import numpy as np
# from parameterized import parameterized
from datasets import load_dataset
from datasets import disable_caching
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
# from open_pref_eval.trainer import dummy_dataset, OPEConfig, OPETrainer
from open_pref_eval.evaluation import evaluate, evaluate_model
from open_pref_eval.helpers.load_model import load_hf_or_peft_model

PEFT_MODELS = [
    "llamafactory/tiny-random-Llama-3-lora",
    # "farpluto/SmolLM-135M-Instruct-Finetune-LoRA",
    # "bunnycore/SmolLM2-1.7B-lora_model",
    # "wassname/qwen-7B-codefourchan-QLoRA",
]

MODELS = [
            "snake7gun/tiny-random-qwen3",
            "HuggingFaceTB/SmolLM2-135M-Instruct",

            # "gepardzik/LLama-3-8b-rogue-lora",
]

N = 80

disable_caching()
imdb = load_dataset('wassname/imdb_preferences', split=f'test[:{N}]', keep_in_memory=False)
datasets = [imdb]

@pytest.mark.parametrize(
    "model_name",
    PEFT_MODELS + MODELS
)
def test_model_load(model_name):
    model, tokenizer = load_hf_or_peft_model(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase), f"Tokenizer {model_name} is not a PreTrainedTokenizerBase"


@pytest.mark.parametrize(
    "model_name",
    PEFT_MODELS
)
def test_evaluate_peft_model( model_name):
    print('testing', model_name)
    model, tokenizer = load_hf_or_peft_model(model_name)

    df, df_raw = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        verbose=3,
    )

    # Only calculate mean for numeric columns
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    print('df_raw', df_raw.groupby(["model", "dataset"], dropna=False)[numeric_cols].mean())
    print(df)
    assert df['correct'].iloc[0]>0.5


@pytest.mark.parametrize(
    "model_name",
    MODELS,
)
def test_evaluate(model_name):
    df_agg, _ = evaluate(
        model_names=[model_name], datasets=datasets,
        batch_size=4,
        verbose=3,
    )
    print(df_agg)
    if 'random' in model_name:
        pass
    else:
        s = df_agg['correct'].iloc[0]
        assert s>0.5, f"Model {model_name} did not perform better than random guessing on the dataset {s}"


@pytest.mark.parametrize(
    "model_name",
    MODELS[:1],
)
def test_datasets(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from open_pref_eval.evaluation import evaluate_model
    from open_pref_eval.data import tokenize_dataset
    from open_pref_eval.datasets import get_default_datasets
    datasets = get_default_datasets(50)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for ds in datasets:
        print(f"Dataset: {ds}")
        tokenize_dataset(ds, tokenizer, verbose=True)

