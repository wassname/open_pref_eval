
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float, Int

EPS = 1e-8

# ============================================================================
# Utility Functions
# ============================================================================

def log_softmax_normalize(log_values: Tensor, mask: Tensor) -> Tensor:
    """Normalize log values using log-softmax, masking out padding tokens."""
    # Mask out padding as -inf so it gets zero weight in softmax
    masked_log_values = log_values.masked_fill(mask == 0, float('-inf'))
    # Subtract logsumexp to normalize in log-space
    return masked_log_values - torch.logsumexp(masked_log_values, dim=-1, keepdim=True)

def prob_normalize(values: Tensor, mask: Tensor) -> Tensor:
    """Normalize probability values, handling masked tokens."""
    # Zero out masked positions, then normalize
    masked_values = values * mask
    return masked_values / (masked_values.sum(-1, keepdim=True) + EPS)

def first_nonzero(x: Float[Tensor, 'b t'], dim: int = 1) -> Float[Tensor, 'b']:
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


def score_agg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], agg=np.min, **kwargs):

    """
    calculate if the chosen completion is higher than the rejected, using the AGG logprob of the sequence. Where agg could be min, max, mean, sum, first, last, etc.
    """
    # get the total logprob of the completion
    c = agg(logp_c * mask_c)
    r = agg(logp_r * mask_r)

    # and the ratio in logspace
    logratio = c - r

    return out(c, r)


def first_nonzero(x: Float[Tensor, 'b t'], dim=1) -> Float[Tensor, 'b']:
    """get the first non zero element in a tensor"""
    return x[torch.arange(x.shape[0]), (x != 0).float().argmax(dim=dim)]

def score_1st_diverg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """
    calculate if the chosen completion is higher than the rejected, using first divering token. This gives a contrasting signal, but it's noisy.

    """
    c = first_nonzero(logp_c * mask_c)
    r = first_nonzero(logp_r * mask_r)
    return out(c, r)

def score_preferences(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """
    calculate if the chosen completion is higher than the rejected, using DPO

    this has a problem is one string is longer than the other, or when some token are very low prob
    """
    # get the total logprob of the completion
    # maxl = min(logp_c.shape[1], logp_r.shape[1]) # FIXME need to ignore padding
    maxl = max(mask_c.sum(-1).max(), mask_r.sum(-1).max()).long()
    c = (logp_c * mask_c)[:, :maxl].sum(-1)
    r = (logp_r * mask_r)[:, :maxl].sum(-1)

    return out(c, r)


def score_ipo(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """
    IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
    Unlike preference loss which takes the combined prob of the whole sequence, this takes the mean prob of each token. This gives a pretty good signal.
    """
    # get the avg logprob of the completion
    c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1).clamp(min=eps)
    r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1).clamp(min=eps)

    return out(c, r)


def score_cumsum(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    # get the avg logprob over the cumulative logprob of each token, this means the initial tokens are weighted higher, but all tokens have an influenceeps = eps
    cumsum_c = (logp_c * mask_c).cumsum(-1)
    # Weight by position (early tokens get higher weight)
    c = (cumsum_c * mask_c).sum(-1) / (mask_c.sum(-1) + eps)

    cumsum_r = (logp_r * mask_r).cumsum(-1)
    r = (cumsum_r * mask_r).sum(-1) / (mask_r.sum(-1) + eps)

    return out(c, r)


def score_weighted(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """Weight by cumulative logprob."""
    
    logp_cm = (logp_c * mask_c)
    logp_rm = (logp_r * mask_r)

    # get weights
    logp_cmcs = logp_cm.cumsum(-1) * mask_c
    logp_rmcs = logp_rm.cumsum(-1) * mask_r

    c_w = log_norm(logp_cmcs, mask_c) * mask_c
    r_w = log_norm(logp_rmcs, mask_r) * mask_r

    cs = (logp_cm * c_w * mask_c).nansum(-1) / (c_w.nansum(-1) + eps)
    rs = (logp_rm * r_w * mask_r).nansum(-1) / (r_w.nansum(-1) + eps)

    # FIXME this is somehow coming out -12 to 12
    return out(cs, rs)

def score_weighted_prob(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """Weight by cumulative prob."""
    
   # Apply exp to log probabilities and mask
    p_c = torch.exp(logp_c) * mask_c
    p_r = torch.exp(logp_r) * mask_r

    # Calculate cumulative sum of probabilities
    cum_p_c = torch.cumprod(p_c, dim=1)
    cum_p_r = torch.cumprod(p_r, dim=1)

    cum_p_c = norm(cum_p_c, mask_c) * mask_c
    cum_p_r = norm(cum_p_r, mask_r) * mask_r

    # Calculate weighted mean
    cs = torch.sum(p_c * cum_p_c * mask_c, dim=1) / (cum_p_c.sum(-1) + eps)
    rs = torch.sum(p_r * cum_p_r * mask_r, dim=1) / (cum_p_r.sum(-1) + eps)

    # return uncalibrated probability
    return out(cs.log(), rs.log())

    
def score_with_entropy_weight(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r, alpha=1, **kwargs):
    # here we downweight uncertain tokens. E.g. if it's low prob because everything is low, we want to reduce its impact
    # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
    # https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L1197
    logp_c_adjusted = logp_c - logp_vocab_conc_c * alpha
    logp_c_w = (logp_c_adjusted * mask_c).sum(-1) / mask_c.sum(-1).clamp(min=eps)
    logp_r_adjusted = logp_r - logp_vocab_conc_r * alpha
    logp_r_w = (logp_r_adjusted * mask_r).sum(-1) / mask_r.sum(-1).clamp(min=eps)
    return out(logp_c_w, logp_r_w)

def score_confidence_weighted(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r, T=1.0, **kwargs):
    # Convert concentration to confidence weights (higher concentration = lower confidence)
    # Use softmax to normalize across sequence
    conf_c = torch.softmax(-logp_vocab_conc_c/T, dim=-1) * mask_c * T
    conf_r = torch.softmax(-logp_vocab_conc_r/T, dim=-1) * mask_r * T

    # Weight the log probabilities by confidence
    weighted_logp_c = (logp_c * conf_c * mask_c).sum(-1) / (conf_c * mask_c).sum(-1).clamp(min=eps)
    weighted_logp_r = (logp_r * conf_r * mask_r).sum(-1) / (conf_r * mask_r).sum(-1).clamp(min=eps)
    
    return out(weighted_logp_c, weighted_logp_r)

def score_uncertainty_aware(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r, **kwargs):
    # Treat concentration as variance in a Gaussian approximation
    # Higher concentration = higher uncertainty
    uncertainty_c = logp_vocab_conc_c * mask_c
    uncertainty_r = logp_vocab_conc_r * mask_r
    
    # Precision (inverse variance) weighting
    precision_c = torch.exp(-uncertainty_c) * mask_c
    precision_r = torch.exp(-uncertainty_r) * mask_r
    
    # Precision-weighted mean
    logp_c_weighted = (logp_c * precision_c).sum(-1) / (precision_c.sum(-1) + eps)
    logp_r_weighted = (logp_r * precision_r).sum(-1) / (precision_r.sum(-1) + eps)
    
    return out(logp_c_weighted, logp_r_weighted)

def score_information_weighted(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r, **kwargs):
    # Information content: negative log probability
    # But we want to weight by how informative each token is
    # Higher concentration = less informative
    
    # Convert to "informativeness" (inverse of concentration)
    max_conc = max(logp_vocab_conc_c.max(), logp_vocab_conc_r.max())
    info_c = (max_conc - logp_vocab_conc_c) * mask_c
    info_r = (max_conc - logp_vocab_conc_r) * mask_r
    
    # Normalize to get weights
    info_weights_c = torch.softmax(info_c, dim=-1) * mask_c
    info_weights_r = torch.softmax(info_r, dim=-1) * mask_r
    
    # Information-weighted average
    logp_c_weighted = (logp_c * info_weights_c).sum(-1) / (info_weights_c.sum(-1) + eps)
    logp_r_weighted = (logp_r * info_weights_r).sum(-1) / (info_weights_r.sum(-1) + eps)
    
    return out(logp_c_weighted, logp_r_weighted)

def score_f_divergance(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """
    https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L943
    """
    logp_c = logp_c * mask_c
    logp_r = logp_r * mask_r
    logits_c = logp_c - F.softplus(logp_c)
    logits_c = (logits_c * mask_c).sum(-1) / (mask_c.sum(-1) + eps)

    logits_r = (logp_r - F.softplus(logp_r)) * mask_r
    logits_r = logits_r.sum(-1) / (mask_r.sum(-1) + eps)

    # Calculate weighted sum
    return out(logits_c, logits_r)



def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow.
    The formula is : log(value.dtype.max)
    E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.
    ```

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap.
            eg: direct calling exp(log(torch.float32.max)) will result in inf
            so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def cap_exp(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


def score_f_alpha_divergance(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], **kwargs):
    """
    https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L914
    """
    logp_c *= mask_c
    logp_r *= mask_r
    alpha_coef = .5
    c = cap_exp(logp_c * -alpha_coef)/ alpha_coef
    r = cap_exp(logp_r * -alpha_coef)/ alpha_coef
    c = (c * mask_c).sum(-1) / (mask_c.sum(-1) + eps)
    r = (r * mask_r).sum(-1) / (mask_r.sum(-1) + eps)

    return out(c.log(), r.log())

def score_with_decay(logp_c, logp_r, mask_c, mask_r, decay=0.90, **kwargs):
    # Current code uses indices as exponent, should use them as power
    seq_len = logp_c.shape[1]
    decay_weights = torch.pow(decay, torch.arange(seq_len, device=logp_c.device))
    decay_weights = decay_weights.unsqueeze(0)
    
    # Apply decay and mask
    c = (logp_c * mask_c * decay_weights).sum(-1) / (mask_c * decay_weights).sum(-1).clamp(min=eps)
    r = (logp_r * mask_r * decay_weights).sum(-1) / (mask_r * decay_weights).sum(-1).clamp(min=eps)
    
    return out(c, r)


def score_power_mean(logp_c, logp_r, mask_c, mask_r, p=0.5, **kwargs):
    # p=0: geometric mean, p=1: arithmetic mean
    # p=0.5: intermediate
    pc = torch.exp(logp_c) * mask_c
    pr = torch.exp(logp_r) * mask_r
    
    score_c = (pc.pow(p).sum(-1) / mask_c.sum(-1).clamp(min=eps)).pow(1/p)
    score_r = (pr.pow(p).sum(-1) / mask_r.sum(-1).clamp(min=eps)).pow(1/p)
    return out(score_c.log(), score_r.log())


def score_seq_entropy_weighted(logp_c, logp_r, mask_c, mask_r, **kwargs):
    # Need to handle zeros in p to avoid nan
    p_c = torch.exp(logp_c) * mask_c + eps
    p_r = torch.exp(logp_r) * mask_r + eps

    # Normalize to valid probability distribution
    p_c = p_c / p_c.sum(-1, keepdim=True)
    p_r = p_r / p_r.sum(-1, keepdim=True)
    
    # Calculate entropy
    entropy_c = -(p_c * torch.log(p_c + eps)).sum(-1)
    entropy_r = -(p_r * torch.log(p_r + eps)).sum(-1)

    # Return difference (not through out() since this isn't a log-prob)
    return out(entropy_c, entropy_r)

# Or weight tokens by their "certainty"
def score_certainty_weighted(logp_c, logp_r, mask_c, mask_r, **kwargs):
    # High prob tokens get more weight
    weights_c = torch.exp(logp_c) * mask_c  
    weights_r = torch.exp(logp_r) * mask_r
    
    # Certainty-weighted average
    score_c = (logp_c * weights_c).sum(-1) / weights_c.sum(-1).clamp(min=eps)
    score_r = (logp_r * weights_r).sum(-1) / weights_r.sum(-1).clamp(min=eps)
    
    return out(score_c, score_r)


def score_percentile(logp_c, logp_r, mask_c, mask_r, percentile=75, **kwargs):
    # Replace padded values with nan, then use nanquantile
    logp_c_masked = logp_c.clone()
    logp_r_masked = logp_r.clone()
    
    logp_c_masked[~mask_c.bool()] = float('nan')
    logp_r_masked[~mask_r.bool()] = float('nan')
    
    #hmm length adjust?
    maxl = max(mask_c.sum(-1).max(), mask_r.sum(-1).max()).long()
    score_c = torch.nanquantile(logp_c_masked.float()[:, :maxl], percentile/100., dim=1)
    score_r = torch.nanquantile(logp_r_masked.float()[:, :maxl], percentile/100., dim=1)

    return out(score_c, score_r)

def score_perplexity_ratio(logp_c, logp_r, mask_c, mask_r, **kwargs):
    # Calculate perplexities
    avg_logp_c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1).clamp(min=eps)
    avg_logp_r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1).clamp(min=eps)
    
    perp_c = torch.exp(-avg_logp_c)
    perp_r = torch.exp(-avg_logp_r)
    
    # Lower perplexity is better, so r/c gives >1 if chosen is better
    return out(perp_r.log(), perp_c.log())
    return perp_c / (perp_c + perp_r)
    return out(perp_r.log(), perp_c.log())



def score_rank(mask_c, mask_r, chosen_ranks, rejected_ranks, **kwargs):
    """
    Score a sequence by the rank (not logprob)
    """
    # Need to handle zeros in p to avoid nan
    p_c = (torch.log(chosen_ranks) * mask_c) / (mask_c.sum(dim=-1, keepdim=True) + eps)
    p_r = (torch.log(rejected_ranks) * mask_r) / (mask_r.sum(dim=-1, keepdim=True) + eps)

    # Return difference (not through out() since this isn't a log-prob)
    return out(p_c, p_r)
