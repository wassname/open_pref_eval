import tempfile
import unittest

import pytest
import numpy as np
# from parameterized import parameterized
from datasets import load_dataset
from datasets import disable_caching

# from open_pref_eval.trainer import dummy_dataset, OPEConfig, OPETrainer
from open_pref_eval.evaluation import evaluate, evaluate_model
from open_pref_eval.helpers.peft_utils import load_hf_or_peft_model

PEFT_MODELS = [
    "pacozaa/tinyllama-alpaca-lora", # 1.1b
]

MODELS = [
            # https://huggingface.co/models?other=base_model%3Aadapter%3Aunsloth%2Ftinyllama
            # "pacozaa/tinyllama-alpaca-lora", # 1.1b
            # "snake7gun/tiny-random-qwen3",
            "HuggingFaceTB/SmolLM2-135M-Instruct",

            # ["gepardzik/LLama-3-8b-rogue-lora"],
            # ["t5"], # TODO make it work for encoder_decoder
]

N = 80

disable_caching()
imdb = load_dataset('wassname/imdb_preferences', split=f'test[:{N}]', keep_in_memory=False)
datasets = [imdb]

@pytest.mark.parametrize(
    "model_name",
    PEFT_MODELS+ MODELS
)
def test_loading(model_name):
    model, tokenizer = load_hf_or_peft_model(model_name)


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
    )

    # Only calculate mean for numeric columns
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    print('df_raw', df_raw.groupby(["model", "dataset"], dropna=False)[numeric_cols].mean())
    print(df)
    assert df['correct'].iloc[0]>0.5


@pytest.mark.parametrize(
    "model_name",
    MODELS[:1],
)
def test_evaluate(model_name):
    df_agg, _ = evaluate(
        model_names=[model_name], datasets=datasets,
        batch_size=4,
        )
    print(df_agg)
    assert df_agg['correct'].iloc[0]>0.5


def test_datasets(dataset):
    model_name = MODELS[0]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from open_pref_eval.evaluation import evaluate_model
    from open_pref_eval.helpers.tokenize import tokenize_dataset
    from open_pref_eval.helpers.datasets import get_default_datasets
    datasets = get_default_datasets(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for ds in datasets:
        print(f"Dataset: {ds}")
        tokenize_dataset(ds, tokenizer, verbose=True)

