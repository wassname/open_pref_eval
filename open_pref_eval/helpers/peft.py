from contextlib import contextmanager, nullcontext
from trl.import_utils import is_peft_available

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def is_peft_model(model):
    if is_peft_available() and isinstance(model, PeftModel):
        return True
    return False


@contextmanager
def set_adapter(model: PeftModel, adapter_name: str = None):
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
    finally:
        model.set_adapter(old_adapter_name)
