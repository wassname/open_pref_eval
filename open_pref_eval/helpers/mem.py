import gc
import torch

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
