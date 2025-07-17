import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float

EPS = 1e-8

# ============================================================================
# Core Utility Functions
# ============================================================================

def build_output_dict(chosen_log_score: Float[Tensor, "batch seq_len"], rejected_log_score: Float[Tensor, "batch seq_len"]) -> dict:
    """Build standardized output dictionary from log scores."""
    log_ratio = chosen_log_score - rejected_log_score
    return {
        'sigmoid': torch.sigmoid(log_ratio),
        'log_ratio': log_ratio,
        'correct': chosen_log_score > rejected_log_score,
        'chosen_log_score': chosen_log_score,
        'rejected_log_score': rejected_log_score,
    }

def log_softmax_normalize(log_values: Tensor, mask: Tensor) -> Tensor:
    """Normalize log values using log-softmax, masking out padding tokens."""
    masked_log_values = log_values.masked_fill(mask == 0, float('-inf'))
    return masked_log_values - torch.logsumexp(masked_log_values, dim=-1, keepdim=True)

def prob_normalize(values: Tensor, mask: Tensor) -> Tensor:
    """Normalize probability values, handling masked tokens."""
    masked_values = values * mask
    return masked_values / (masked_values.sum(-1, keepdim=True) + EPS)

def first_nonzero_index(x: Tensor, dim: int = 1) -> Tensor:
    """Get the first non-zero element in a tensor along the specified dimension."""
    return x[torch.arange(x.shape[0]), (x != 0).float().argmax(dim=dim)]

def get_exp_cap(value: Tensor, decimal: int = 4) -> Tensor:
    """Get the exponent cap to avoid overflow when calling torch.exp."""
    dtype_max = torch.finfo(value.dtype).max
    log_max = torch.log(torch.tensor(dtype_max, dtype=value.dtype, device=value.device))
    return torch.floor(log_max * 10**decimal) / 10**decimal if decimal > 0 else log_max

def safe_exp(log_values: Tensor, cap: float = -1) -> Tensor:
    """Safely compute exp with clamping to avoid overflow."""
    if cap < 0:
        cap = get_exp_cap(log_values)
    return torch.exp(torch.clamp(log_values, max=cap))

# ============================================================================
# Basic Scoring Functions
# ============================================================================

def score_dpo(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using sum of log probabilities (standard DPO)."""
    # Sum log probabilities over sequence length, respecting masks
    chosen_score = (log_prob_chosen * mask_chosen).sum(-1)
    rejected_score = (log_prob_rejected * mask_rejected).sum(-1)
    
    return build_output_dict(chosen_score, rejected_score)

def score_ipo(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using mean log probabilities (IPO-style)."""
    chosen_score = (log_prob_chosen * mask_chosen).sum(-1) / mask_chosen.sum(-1).clamp(min=EPS)
    rejected_score = (log_prob_rejected * mask_rejected).sum(-1) / mask_rejected.sum(-1).clamp(min=EPS)
    
    return build_output_dict(chosen_score, rejected_score)

def score_uln(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    log_prob_chosen_norm: Tensor,
    log_prob_rejected_norm: Tensor,
    **kwargs
) -> dict:
    """
    Score using unconditional likelihood normalized (ULN).

    Unconditional likelihood normalized selects the option with the highest average token logprob once normalized by the unconditional token logprobs, as described in this EleutherAI blogpost. This method incurs an additional LLM call to obtain the unconditional likelihoods.

    https://blog.eleuther.ai/multiple-choice-normalization/
    """
    chosen_score = (log_prob_chosen_norm * mask_chosen).sum(-1) / mask_chosen.sum(-1).clamp(min=EPS)
    rejected_score = (log_prob_rejected_norm * mask_rejected).sum(-1) / mask_rejected.sum(-1).clamp(min=EPS)

    return build_output_dict(chosen_score, rejected_score)
