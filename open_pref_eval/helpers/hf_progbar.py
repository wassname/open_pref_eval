import contextlib
from datasets.utils.logging import disable_progress_bar, enable_progress_bar, is_progress_bar_enabled

@contextlib.contextmanager
def no_hf_tqdm():
    """turn off the huggingface datasets progress bar"""
    original_value = is_progress_bar_enabled()
    disable_progress_bar()
    try:
        yield
    finally:
        if original_value:
            enable_progress_bar()
