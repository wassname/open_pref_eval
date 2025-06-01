import tempfile
import unittest

import pytest
# from parameterized import parameterized
from datasets import load_dataset


# from open_pref_eval.trainer import dummy_dataset, OPEConfig, OPETrainer
from open_pref_eval.evaluation import evaluate, evaluate_model
from open_pref_eval.helpers.load_models import load_peft_model

MODELS = [
            # https://huggingface.co/models?other=base_model%3Aadapter%3Aunsloth%2Ftinyllama
            "pacozaa/tinyllama-alpaca-lora", # 101mb
            "snake7gun/tiny-random-qwen3",
            "HuggingFaceTB/SmolLM2-135M-Instruct",

            # ["gepardzik/LLama-3-8b-rogue-lora"],
            "bunnycore/Phi-3.5-mini-lora-rp",
            # ["tloen/alpaca-lora-7b"],
            # ["t5"], # TODO make it work for encoder_decoder
]

N = 80
imdb = load_dataset('wassname/imdb_preferences', split=f'test[:{N}]', keep_in_memory=False)
datasets = [imdb]


@pytest.mark.parametrize(
    "model_name",
    MODELS
)
def test_evaluate_model( model_name):
    print('testing', model_name)
    model, tokenizer = load_peft_model(model_name)

    df, df_raw = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
    )

    print('df_raw', df_raw.groupby(["model", "dataset"], dropna=False).mean())
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
