import tempfile
import unittest

import pytest
import torch
from datasets import Dataset, features
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from trl import DPOConfig, DPOTrainer

from open_pref_eval.trainer import get_dummy_trainer, dummy_dataset
from open_pref_eval.evaluation import evaluate, eval_dpo_dataset



class DPOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/T5ForConditionalGeneration-correct-vocab-calibrated"
        cls.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # get idefics2 model
        model_id = "trl-internal-testing/tiny-random-idefics2"
        cls.idefics2_model = AutoModelForVision2Seq.from_pretrained(model_id)
        cls.idefics2_ref_model = AutoModelForVision2Seq.from_pretrained(model_id)
        cls.idefics2_processor = AutoProcessor.from_pretrained(model_id)

def test_dpo_trainer(self, name, loss_type, pre_compute):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            df = evaluate(trainer=trainer, datasets=[dummy_dataset])

            assert len(df)>0
