from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available
from transformers import PreTrainedModel

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

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
        print(f"Error: {e}")
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
        print(f"Error: {e}")
        raise e
    finally:
        if old_adapter_name is None:
            model.disable_adapter()
        else:
            model.set_adapter(old_adapter_name)
