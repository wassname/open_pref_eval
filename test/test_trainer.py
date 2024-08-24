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



class DPOTrainerTester(unittest.TestCase):

    @parameterized.expand(
        [
            ["gpt2"],
            # ["t5"], # TODO make it work for encoder_decoder
        ]
    )
    def test_dpo_trainer(self, name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OPEConfig(
                output_dir=tmp_dir,
                per_device_eval_batch_size=2,
            )

            if name == "gpt2":
                model = self.model
                ref_model = self.ref_model
                tokenizer = self.tokenizer
            elif name == "t5":
                model = self.t5_model
                ref_model = self.t5_ref_model
                tokenizer = self.t5_tokenizer

            trainer = OPETrainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
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
        [
            ["gpt2"],
            # ["t5"], # TODO make it work for encoder_decoder
        ]
    )
    def test_evaluate(self, name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = evaluate(
                model_names=[name], datasets=[dummy_dataset],
                batch_size=4,

                # output_dir=tmp_dir,
                )
            print(df)
            assert df['correct'].iloc[0]>0.5


if __name__ == '__main__':
    unittest.main()
