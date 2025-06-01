import torch
from torch import Tensor
import torch.nn.functional as F

EPS = 1e-8

# ============================================================================
# Core Utility Functions
# ============================================================================

def build_output_dict(chosen_log_score: Tensor, rejected_log_score: Tensor) -> dict:
    """Build standardized output dictionary from log scores."""
    log_ratio = chosen_log_score - rejected_log_score
    return {
        'sigmoid': torch.sigmoid(log_ratio),
        'log_ratio': log_ratio,
        'correct': chosen_log_score > rejected_log_score,
        'chosen_log_score': chosen_log_score,
        'rejected_log_score': rejected_log_score,
    }

# def log_softmax_normalize(log_values: Tensor, mask: Tensor) -> Tensor:
#     """Normalize log values using log-softmax, masking out padding tokens."""
#     masked_log_values = log_values.masked_fill(mask == 0, float('-inf'))
#     return masked_log_values - torch.logsumexp(masked_log_values, dim=-1, keepdim=True)

# def prob_normalize(values: Tensor, mask: Tensor) -> Tensor:
#     """Normalize probability values, handling masked tokens."""
#     masked_values = values * mask
#     return masked_values / (masked_values.sum(-1, keepdim=True) + EPS)

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

def score_log_prob_sum(
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

def score_log_prob_mean(
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

def score_first_diverging_token(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using first non-zero (diverging) token log probability."""
    chosen_score = first_nonzero_index(log_prob_chosen * mask_chosen)
    rejected_score = first_nonzero_index(log_prob_rejected * mask_rejected)
    
    return build_output_dict(chosen_score, rejected_score)

def score_position_weighted(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    decay: float = 0.9,
    **kwargs
) -> dict:
    """Score with exponential position decay (earlier tokens matter more)."""
    seq_len = log_prob_chosen.shape[1]
    decay_weights = torch.pow(decay, torch.arange(seq_len, device=log_prob_chosen.device)).unsqueeze(0)
    
    chosen_score = (log_prob_chosen * mask_chosen * decay_weights).sum(-1) / (mask_chosen * decay_weights).sum(-1).clamp(min=EPS)
    rejected_score = (log_prob_rejected * mask_rejected * decay_weights).sum(-1) / (mask_rejected * decay_weights).sum(-1).clamp(min=EPS)
    
    return build_output_dict(chosen_score, rejected_score)

def score_cumulative_weighted(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score weighted by cumulative log probability."""
    cumsum_chosen = (log_prob_chosen * mask_chosen).cumsum(-1)
    cumsum_rejected = (log_prob_rejected * mask_rejected).cumsum(-1)
    
    chosen_score = (cumsum_chosen * mask_chosen).sum(-1) / (mask_chosen.sum(-1) + EPS)
    rejected_score = (cumsum_rejected * mask_rejected).sum(-1) / (mask_rejected.sum(-1) + EPS)
    
    return build_output_dict(chosen_score, rejected_score)

# ============================================================================
# Statistical Scoring Functions
# ============================================================================

def score_percentile(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    percentile: float = 75,
    **kwargs
) -> dict:
    """Score using percentile of log probabilities."""
    # Replace masked tokens with NaN for proper percentile calculation
    log_prob_chosen_masked = log_prob_chosen.clone()
    log_prob_rejected_masked = log_prob_rejected.clone()
    
    log_prob_chosen_masked[~mask_chosen.bool()] = float('nan')
    log_prob_rejected_masked[~mask_rejected.bool()] = float('nan')
    
    chosen_score = torch.nanquantile(log_prob_chosen_masked.float(), percentile/100., dim=1)
    if torch.isnan(chosen_score).any():
        chosen_score = torch.nan_to_num(chosen_score, nan=0.0)
    rejected_score = torch.nanquantile(log_prob_rejected_masked.float(), percentile/100., dim=1)
    if torch.isnan(rejected_score).any():
        rejected_score = torch.nan_to_num(rejected_score, nan=0.0)
    
    return build_output_dict(chosen_score, rejected_score)

def score_power_mean(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    power: float = 0.5,
    **kwargs
) -> dict:
    """Score using power mean of probabilities (p=0: geometric, p=1: arithmetic)."""
    prob_chosen = torch.exp(log_prob_chosen) * mask_chosen
    prob_rejected = torch.exp(log_prob_rejected) * mask_rejected
    
    chosen_score = (prob_chosen.pow(power).sum(-1) / mask_chosen.sum(-1).clamp(min=EPS)).pow(1/power)
    rejected_score = (prob_rejected.pow(power).sum(-1) / mask_rejected.sum(-1).clamp(min=EPS)).pow(1/power)
    
    return build_output_dict(chosen_score.log(), rejected_score.log())

def score_entropy_weighted(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using sequence entropy."""
    prob_chosen = torch.exp(log_prob_chosen) * mask_chosen + EPS
    prob_rejected = torch.exp(log_prob_rejected) * mask_rejected + EPS
    
    # Normalize to valid probability distributions
    prob_chosen = prob_chosen / prob_chosen.sum(-1, keepdim=True)
    prob_rejected = prob_rejected / prob_rejected.sum(-1, keepdim=True)
    
    entropy_chosen = -(prob_chosen * torch.log(prob_chosen + EPS)).sum(-1)
    entropy_rejected = -(prob_rejected * torch.log(prob_rejected + EPS)).sum(-1)
    
    return build_output_dict(entropy_chosen, entropy_rejected)

def score_confidence_weighted(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score weighted by token confidence (high probability tokens get more weight)."""
    prob_chosen = torch.exp(log_prob_chosen) * mask_chosen
    prob_rejected = torch.exp(log_prob_rejected) * mask_rejected
    
    chosen_score = (log_prob_chosen * prob_chosen).sum(-1) / prob_chosen.sum(-1).clamp(min=EPS)
    rejected_score = (log_prob_rejected * prob_rejected).sum(-1) / prob_rejected.sum(-1).clamp(min=EPS)
    
    return build_output_dict(chosen_score, rejected_score)


# ============================================================================
# Divergence-Based Scoring Functions
# ============================================================================

def score_f_divergence(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using f-divergence (from TRL)."""
    masked_log_prob_chosen = log_prob_chosen * mask_chosen
    masked_log_prob_rejected = log_prob_rejected * mask_rejected
    
    logits_chosen = masked_log_prob_chosen - F.softplus(masked_log_prob_chosen)
    logits_rejected = masked_log_prob_rejected - F.softplus(masked_log_prob_rejected)
    
    chosen_score = (logits_chosen * mask_chosen).sum(-1) / (mask_chosen.sum(-1) + EPS)
    rejected_score = (logits_rejected * mask_rejected).sum(-1) / (mask_rejected.sum(-1) + EPS)
    
    return build_output_dict(chosen_score, rejected_score)

def score_alpha_divergence(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    alpha: float = 0.5,
    **kwargs
) -> dict:
    """Score using alpha-divergence."""
    masked_log_prob_chosen = log_prob_chosen * mask_chosen
    masked_log_prob_rejected = log_prob_rejected * mask_rejected
    
    chosen_exp = safe_exp(masked_log_prob_chosen * -alpha) / alpha
    rejected_exp = safe_exp(masked_log_prob_rejected * -alpha) / alpha
    
    chosen_score = (chosen_exp * mask_chosen).sum(-1) / (mask_chosen.sum(-1) + EPS)
    rejected_score = (rejected_exp * mask_rejected).sum(-1) / (mask_rejected.sum(-1) + EPS)

    return build_output_dict(chosen_score.clamp(min=EPS).log(), rejected_score.clamp(min=EPS).log())

# ============================================================================
# Alternative Scoring Functions
# ============================================================================

def score_perplexity_ratio(
    log_prob_chosen: Tensor, 
    log_prob_rejected: Tensor, 
    mask_chosen: Tensor, 
    mask_rejected: Tensor, 
    **kwargs
) -> dict:
    """Score using perplexity ratio (lower perplexity is better)."""
    avg_log_prob_chosen = (log_prob_chosen * mask_chosen).sum(-1) / mask_chosen.sum(-1).clamp(min=EPS)
    avg_log_prob_rejected = (log_prob_rejected * mask_rejected).sum(-1) / mask_rejected.sum(-1).clamp(min=EPS)
    
    # Convert to perplexity (lower is better)
    perp_chosen = torch.exp(-avg_log_prob_chosen)
    perp_rejected = torch.exp(-avg_log_prob_rejected)
    
    # Return inverted scores so lower perplexity = higher score
    return build_output_dict(perp_rejected.log(), perp_chosen.log())

# ============================================================================
# Legacy Function Aliases (for backward compatibility)
# ============================================================================

def score_preferences(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs):
    """Legacy alias for score_log_prob_sum."""
    return score_log_prob_sum(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs)

def score_ipo(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs):
    """Legacy alias for score_log_prob_mean."""
    return score_log_prob_mean(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs)

def score_1st_diverg(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs):
    """Legacy alias for score_first_diverging_token."""
    return score_first_diverging_token(log_prob_chosen, log_prob_rejected, mask_chosen, mask_rejected, **kwargs)
