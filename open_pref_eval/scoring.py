
import torch
from torch import Tensor
from typing import Optional, List, Union
from collections import OrderedDict
import numpy as np
from jaxtyping import Float, Int


def first_nonzero(x: Float[Tensor, 'b t'], dim=1) -> Float[Tensor, 'b']:
    """get the first non zero element in a tensor"""
    return x[torch.arange(x.shape[0]), (x != 0).float().argmax(dim=dim)]

def score_1st_diverg(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    """
    calculate if the chosen completion is higher than the rejected, using first divering token

    return uncalibrated probability
    """
    m = mask_c * mask_r
    logratio = (logp_c - logp_r) * m
    return torch.sigmoid(first_nonzero(logratio))

def score_preferences(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    """
    calculate if the chosen completion is higher than the rejected, using DPO

    return uncalibrated probability
    """
    # get the total logprob of the completion
    c = (logp_c * mask_c).sum(-1)
    r = (logp_r * mask_r).sum(-1)

    # and the ratio in logspace
    logratio = c - r

    # return uncalibrated probability
    return torch.sigmoid(logratio)


def score_ipo(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    # get the avg logprob of the completion
    c = (logp_c * mask_c).sum(-1) / mask_c.sum(-1)
    r = (logp_r * mask_r).sum(-1) / mask_r.sum(-1)

    # and the ratio in logspace
    logratio = c - r

    # return uncalibrated probability
    return torch.sigmoid(logratio)


def score_cumsum(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    # get the avg logprob over the cumulative logprob of each token, this means the initial tokens are weighted higher, but all tokens have an influence
    c = (logp_c * mask_c).cumsum(-1)
    c = (c * mask_c).sum(-1) / mask_c.sum(-1)
    r = (logp_r * mask_r).cumsum(-1)
    r = (r * mask_r).sum(-1) / mask_r.sum(-1)

    # and the ratio in logspace
    logratio = c - r

    # return uncalibrated probability
    return torch.sigmoid(logratio)


def score_weighted(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    """Calc score, this time it's the weighted sum of the logprobs. We weight by cumulative prob."""
    eps = 1e-8
    
    c = (logp_c * mask_c)
    r = (logp_r * mask_r)

    # get weights
    cc = c.cumsum(-1) * mask_c
    rr = r.cumsum(-1) * mask_r

    def norm(x):
        x = x - x.min(-1, keepdim=True)[0]
        return x / (x.sum(-1, keepdim=True)+eps)
    
    c_w = norm(cc)
    r_w = norm(rr)

    # weighted mean
    cs = (c * c_w).sum(-1) / (c_w.sum(-1) + eps)
    rs = (r * r_w).sum(-1) / (r_w.sum(-1) + eps)

    # and the ratio in logspace
    logratio = cs - rs

    # return uncalibrated probability
    return torch.sigmoid(logratio)

def score_weighted_prob(logp_c: Float[Tensor, 'b t'], logp_r: Float[Tensor, 'b t'], mask_c: Int[Tensor, 'b t'], mask_r: Int[Tensor, 'b t']):
    """Calc score, this time it's the weighted sum of the logprobs. We weight by cumulative prob."""
    eps = 1e-8
    
   # Apply exp to log probabilities and mask
    p_c = torch.exp(logp_c) * mask_c
    p_r = torch.exp(logp_r) * mask_r

    # Calculate cumulative sum of probabilities
    cum_p_c = torch.cumprod(p_c, dim=1)
    cum_p_r = torch.cumprod(p_r, dim=1)

    # Normalize weights
    def norm(x):
        x = x - x.min(-1, keepdim=True)[0]
        return x / (x.sum(-1, keepdim=True)+eps)
    cum_p_c = norm(cum_p_c) * mask_c
    cum_p_r = norm(cum_p_r) * mask_r

    # Calculate weighted sum
    cs = torch.sum(p_c * cum_p_c, dim=1) / (cum_p_c.sum(-1) + eps)
    rs = torch.sum(p_r * cum_p_r, dim=1) / (cum_p_r.sum(-1) + eps)

    # return uncalibrated probability
    return cs / (cs + rs + eps)
