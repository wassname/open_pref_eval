from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available
from transformers import PreTrainedModel
from loguru import logger
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .peft_utils import is_peft_model_name

if is_peft_available():
    from peft import AutoPeftModelForCausalLM, get_peft_model, PeftConfig, PeftModelForCausalLM
    from peft import PeftModel, get_peft_model

def load_peft_model(adapter_model_name, **model_kwargs):
    if is_peft_available() and is_peft_model_name(adapter_model_name):
        peft_config = PeftConfig.from_pretrained(adapter_model_name)
        base_model_name = peft_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )

        # Note PEFT models can be loaded through PEFT of Transformers, and they have frustratingly similar but different APIs. Best to use PEFT itself.
        model = get_peft_model(model, peft_config)

        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)
        except (OSError, ValueError) as e:
            # sometimes peft models will not define a tokenizer
            logger.exception(f"Failed to load tokenizer for {adapter_model_name}: e:`{e}`. Fallback to base model tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        raise ValueError(f"Model {adapter_model_name} is not a PEFT model.")
    return model, tokenizer

def load_hf_or_peft_model(model_name, load_4bit=False, load_8bit=False, **model_kwargs):

    # Allow quantization via kwargs
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['quantization_config'] = quantization_config
    elif load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs['quantization_config'] = quantization_config
    
    # Handle either PEFT or non-PEFT models
    if is_peft_available() and is_peft_model_name(model_name):
        return load_peft_model(model_name, **model_kwargs)
    else:
        return load_hf_model(model_name, **model_kwargs)
    

def load_hf_model(model_name, **model_kwargs):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        **model_kwargs
    )

    return model, tokenizer
