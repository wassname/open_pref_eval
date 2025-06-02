"""
see also
- https://github.com/JD-P/minihf/blob/9e64b1ffb44c00ebab933301a80b902f422faba4/minihf_infer.py#L37
"""

from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available
from transformers import PreTrainedModel
from loguru import logger
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

if is_peft_available():
    from peft import AutoPeftModelForCausalLM, get_peft_model, PeftConfig, PeftModelForCausalLM
    from peft import PeftModel, get_peft_model

def is_peft_model(model):
    return is_hf_peft_model(model) or is_plain_peft_model(model)

def is_plain_peft_model(model):
    if is_peft_available() and isinstance(model, PeftModel):
        return True
    return False

def is_hf_peft_model(model):
    if is_peft_available() and hasattr(model, 'peft_config'):
        return True
    return False

def adapter_is_disabled(peft_model) -> bool:
    """Given a peft model work out is adapters are enabled or disabled"""
    from peft.peft_model import BaseTunerLayer, PeftModel
    from peft.utils.other import ModulesToSaveWrapper
    for module in peft_model.model.modules():
        if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
            # print(help(module.enable_adapters))
            return module._disable_adapters
        

@contextmanager
def set_adapter(model, adapter_name: str = None):
    """
    Context manager to set the adapter for a model.
    If adapter_name is None, it disables the adapters.
    """
    if is_plain_peft_model(model):
        yield from set_peft_adapter(model, adapter_name)
    elif is_hf_peft_model(model):
        yield from set_hf_adapter(model, adapter_name)
    else:
        raise ValueError("Model is not a PEFT model or HF PEFT model.")

# @contextmanager
def set_hf_adapter(model: PreTrainedModel, adapter_name: str = None):
    old_adapter_name = model.active_adapter()
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            model.disable_adapters()
            yield model
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise e
    finally:
        if old_adapter_name is None:
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_adapter(old_adapter_name)

# @contextmanager
def set_peft_adapter(model: PeftModel, adapter_name: str = None):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise e
    finally:
        if old_adapter_name is None:
            model.disable_adapter()
        else:
            model.set_adapter(old_adapter_name)


def is_peft_model_name(model_name):
    try:
        peft_config = PeftConfig.from_pretrained(model_name)
    except ValueError as e:
        logger.exception(f"Failed to load PeftConfig for {model_name}: {e}")
        return False
    else:
        return True

def load_hf_or_peft_model(model_name, load_4bit=False, load_8bit=False, **model_kwargs):

    # Allow quantization via kwargs
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    
    # Handle either PEFT or non-PEFT models

    if is_peft_available() and is_peft_model_name(model_name):
        # Note PEFT models can be loaded through PEFT of Transformers, and they have frustratingly similar but different APIs. Best to use PEFT itself.
        peft_config = PeftConfig.from_pretrained(model_name)
        adapter_model_name = model_name
        model_name = peft_config.base_model_name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            **model_kwargs
        )
        model = get_peft_model(model, peft_config)

        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)
        except Exception as e:
            logger.exception(f"Failed to load tokenizer for {adapter_model_name}: e:`{e}`. Fallback to base model tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


# TODO test with some of the weird cases I've enountered
if __name__ == "__main__":
    # Example usage
    model_names = [
        "HuggingFaceTB/SmolLM2-135M-Instruct", # no adapter
        "bunnycore/SmolLM2-1.7B-lora_model",
        "Rustamshry/Qwen3-0.6B-OpenMathReason",
        "markab/Qwen1.5-Capybara-0.5B-Chat",
        "gepardzik/LLama-3-8b-rogue-lora",
        "wassname/qwen-7B-codefourchan-QLoRA", # adapter, no tokenizedr?
    ]
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model, tokenizer = load_hf_or_peft_model(model_name, load_4bit=True)
    print(f"Loaded model: {model_name}")
    print(f"Tokenizer: {tokenizer}")
    
    with set_adapter(model, "adapter_name"):
        print("Adapter set.")
    
    print("Done.")
