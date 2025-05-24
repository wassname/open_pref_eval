
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List, Union
from collections import OrderedDict
import numpy as np
from jaxtyping import Float, Int

def log_norm(x, mask):
    # mask out padding as -inf so it gets zero weight
    x = x.masked_fill(mask == 0, float('-inf'))
    # subtract logsumexp to normalize in log-space
    return x - torch.logsumexp(x, dim=-1, keepdim=True)

def norm(x, mask, eps=1e-8):
    # no need to use mask as it's log
    x = x - x.min(-1, keepdim=True)[0]
    return x / (x.sum(-1, keepdim=True)+eps)

def out(clogp, rlogp):
    # max_logp = torch.maximum(clogp, rlogp)
    # exp_c = torch.exp(clogp - max_logp)
    # exp_r = torch.exp(rlogp - max_logp)
    # return exp_c / (exp_c + exp_r)
    return torch.sigmoid((clogp - rlogp))


def score_agg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r, agg=np.min):

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

def score_1st_diverg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """
    calculate if the chosen completion is higher than the rejected, using first divering token. This gives a contrasting signal, but it's noisy.

    """
    m = mask_c * mask_r
    # logratio = (logp_c - logp_r) * m
    # TODO shouldn't all my sigmoid be exp, to get the real ratio?
    # logratio = first_nonzero(logratio)
    # return torch.sigmoid(logratio*100)
    c = first_nonzero(logp_c * mask_c)
    r = first_nonzero(logp_r * mask_r)
    return out(c, r)

def score_preferences(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """
    calculate if the chosen completion is higher than the rejected, using DPO

    this has a problem is one string is longer than the other, or when some token are very low prob
    """
    # get the total logprob of the completion
    # maxl = min(logp_c.shape[1], logp_r.shape[1]) # FIXME need to ignore padding
    maxl = max(mask_c.sum(-1).max(), mask_r.sum(-1).max()).long()
    c = (logp_c * mask_c)[:, :maxl].sum(-1)
    r = (logp_r * mask_r)[:, :maxl].sum(-1)

    # and the ratio in logspace
    logratio = c - r

    # return uncalibrated probability
    # return torch.sigmoid(logratio*100)
    return out(c, r)


def score_ipo(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """
    IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
    Unlike preference loss which takes the combined prob of the whole sequence, this takes the mean prob of each token. This gives a pretty good signal.
    """
    # get the avg logprob of the completion
    c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1)
    r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1)

    # return uncalibrated probability
    # return torch.sigmoid(logratio*100)
    return out(c, r)


def score_cumsum(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    # get the avg logprob over the cumulative logprob of each token, this means the initial tokens are weighted higher, but all tokens have an influence
    eps = 1e-8
    cumsum_c = (logp_c * mask_c).cumsum(-1) # TODO use individual mask
    # Weight by position (early tokens get higher weight)
    c = (cumsum_c * mask_c).sum(-1) / (mask_c.sum(-1) + eps)

    cumsum_r = (logp_r * mask_r).cumsum(-1) # TODO use individual mask
    r = (cumsum_r * mask_r).sum(-1) / (mask_r.sum(-1) + eps)

    # c = (c * mask_c).sum(-1) / mask_c.sum(-1)
    # r = (r * mask_r).sum(-1) / mask_r.sum(-1)

    # and the ratio in logspace
    # logratio = c - r

    # return uncalibrated probability
    # return torch.sigmoid(logratio*100)
    return out(c, r)


def score_weighted(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """Weight by cumulative logprob."""
    eps = 1e-8
    
    logp_cm = (logp_c * mask_c)
    logp_rm = (logp_r * mask_r)

    # get weights
    logp_cmcs = logp_cm.cumsum(-1) * mask_c
    logp_rmcs = logp_rm.cumsum(-1) * mask_r

    c_w = log_norm(logp_cmcs, mask_c) * mask_c
    r_w = log_norm(logp_rmcs, mask_r) * mask_r

    cs = (logp_cm * c_w * mask_c).sum(-1) / (c_w.sum(-1) + eps)
    rs = (logp_rm * r_w * mask_r).sum(-1) / (r_w.sum(-1) + eps)

    return out(cs, rs)

def score_weighted_prob(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """Weight by cumulative prob."""
    eps = 1e-8
    
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

    
def score_with_weight(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    # here we downweight uncertain tokens. E.g. if it's low prob because everything is low, we want to reduce its impact
    # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
    # https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L1197
    logp_c_adjusted = logp_c - logp_vocab_conc_c
    logp_c_w = (logp_c_adjusted * mask_c).sum(-1) / mask_c.sum(-1)
    logp_r_adjusted = logp_r - logp_vocab_conc_r
    logp_r_w = (logp_r_adjusted * mask_r).sum(-1) / mask_r.sum(-1)
    return out(logp_c_w, logp_r_w)

# TODO filter out special tokens, attn_mask, attention sinks



def score_f_divergance(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """
    https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L943
    """
    logp_c = logp_c * mask_c
    logp_r = logp_r * mask_r
    logits_c = logp_c - F.softplus(logp_c)
    logits_c = (logits_c * mask_c).sum(-1) / (mask_c.sum(-1) + 1e-8)

    logits_r = (logp_r - F.softplus(logp_r)) * mask_r
    logits_r = logits_r.sum(-1) / (mask_r.sum(-1) + 1e-8)

    # Calculate weighted sum
    return torch.sigmoid(logits_c - logits_r)



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


def score_f_alpha_divergance(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t'], logp_vocab_conc_c, logp_vocab_conc_r):
    """
    https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L914
    """
    logp_c *= mask_c
    logp_r *= mask_r
    alpha_coef = .5
    c = cap_exp(logp_c * -alpha_coef)/ alpha_coef
    r = cap_exp(logp_r * -alpha_coef)/ alpha_coef
    c = (c * mask_c).sum(-1) / (mask_c.sum(-1) + 1e-8)
    r = (r * mask_r).sum(-1) / (mask_r.sum(-1) + 1e-8)
    logits = c - r

    #logits = torch.sum(logits * mask, dim=1) / (mask.sum(-1) + 1e-8)

    return torch.sigmoid(logits)

def score_with_decay(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r, decay=0.90):
    # Current code uses indices as exponent, should use them as power
    seq_len = logp_c.shape[1]
    decay_weights = torch.pow(decay, torch.arange(seq_len, device=logp_c.device))
    decay_weights = decay_weights.unsqueeze(0)
    
    # Apply decay and mask
    c = (logp_c * mask_c * decay_weights).sum(-1) / (mask_c * decay_weights).sum(-1)
    r = (logp_r * mask_r * decay_weights).sum(-1) / (mask_r * decay_weights).sum(-1)
    
    return out(c, r)


def score_power_mean(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r, p=0.5):
    # p=0: geometric mean, p=1: arithmetic mean
    # p=0.5: intermediate
    pc = torch.exp(logp_c) * mask_c
    pr = torch.exp(logp_r) * mask_r
    
    score_c = (pc.pow(p).sum(-1) / mask_c.sum(-1)).pow(1/p)
    score_r = (pr.pow(p).sum(-1) / mask_r.sum(-1)).pow(1/p)
    return out(score_c.log(), score_r.log())


def score_entropy_weighted(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r):
    # Need to handle zeros in p to avoid nan
    p_c = torch.exp(logp_c) * mask_c + 1e-10
    p_r = torch.exp(logp_r) * mask_r + 1e-10
    
    # Normalize to valid probability distribution
    p_c = p_c / p_c.sum(-1, keepdim=True)
    p_r = p_r / p_r.sum(-1, keepdim=True)
    
    # Calculate entropy
    entropy_c = -(p_c * torch.log(p_c + 1e-10)).sum(-1)
    entropy_r = -(p_r * torch.log(p_r + 1e-10)).sum(-1)
    
    # Return difference (not through out() since this isn't a log-prob)
    return torch.sigmoid(entropy_r - entropy_c)

# Or weight tokens by their "certainty"
def score_certainty_weighted(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r):
    # High prob tokens get more weight
    weights_c = torch.exp(logp_c) * mask_c  
    weights_r = torch.exp(logp_r) * mask_r
    
    # Certainty-weighted average
    score_c = (logp_c * weights_c).sum(-1) / weights_c.sum(-1)
    score_r = (logp_r * weights_r).sum(-1) / weights_r.sum(-1)
    
    return torch.sigmoid(score_c - score_r)


def score_percentile(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r, percentile=75):
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

def score_perplexity_ratio(logp_c, logp_r, mask_c, mask_r, logp_vocab_conc_c, logp_vocab_conc_r):
    # Calculate perplexities
    avg_logp_c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1)
    avg_logp_r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1)
    
    perp_c = torch.exp(-avg_logp_c)
    perp_r = torch.exp(-avg_logp_r)
    
    # Lower perplexity is better, so r/c gives >1 if chosen is better
    return perp_c / (perp_c + perp_r)
    return out(perp_r.log(), perp_c.log())


