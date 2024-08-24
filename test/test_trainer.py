import tempfile
import unittest

import pytest
import torch
from parameterized import parameterized
from datasets import Dataset, features
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)



from open_pref_eval.trainer import get_dummy_trainer, dummy_dataset, OPEConfig, OPETrainer
from open_pref_eval.evaluation import evaluate, eval_dpo_dataset, evaluate_model
from open_pref_eval.helpers.load_models import load_peft_model

MODELS = [
                # ["gpt2"],
            # ["lxe/lora-cerebras-gpt2.7b-alpaca-shortprompt"],
            # ["samadpls/sentiment-analysis"],

            # https://huggingface.co/models?other=base_model%3Aadapter%3Aunsloth%2Ftinyllama
            ["pacozaa/tinyllama-alpaca-lora"],
            ["gepardzik/LLama-3-8b-rogue-lora"],
            ["bunnycore/Phi-3.5-mini-lora-rp"],
            # ["tloen/alpaca-lora-7b"],
            # ["t5"], # TODO make it work for encoder_decoder
]

class DPOTrainerTester(unittest.TestCase):

    @parameterized.expand(
        MODELS
    )
    def test_dpo_trainer(self, model_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model, tokenizer = load_peft_model(model_name)

            training_args = OPEConfig(
                output_dir=tmp_dir,
                per_device_eval_batch_size=2,
            )

            trainer = OPETrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            df, df_raw = evaluate_model(
                trainer=trainer, 
                datasets=[dummy_dataset]
            )

            # TODO assert acc > 0.5 on IMBD sentiment dataset too
            print(df)
            assert df['correct'].iloc[0]>0.5

    @parameterized.expand(
        MODELS[:1],
    )
    def test_evaluate(self, model_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = evaluate(
                model_names=[model_name], datasets=[dummy_dataset],
                batch_size=4,

                # output_dir=tmp_dir,
                )
            print(df)
            assert df['correct'].iloc[0]>0.5


if __name__ == '__main__':
    unittest.main()
