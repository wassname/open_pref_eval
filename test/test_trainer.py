import tempfile
import unittest

import pytest
import numpy as np
# from parameterized import parameterized
from datasets import load_dataset
from datasets import disable_caching

# from open_pref_eval.trainer import dummy_dataset, OPEConfig, OPETrainer
from open_pref_eval.evaluation import evaluate, evaluate_model
from open_pref_eval.helpers.load_models import load_peft_model

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
    PEFT_MODELS
)
def test_evaluate_peft_model( model_name):
    print('testing', model_name)
    model, tokenizer = load_peft_model(model_name)

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
