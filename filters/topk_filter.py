# filters/topk_filter.py

import torch
from .base import BaseFilter, FilterResult


class TopKFilter(BaseFilter):
    """
    No-control baseline. Just selects top-k tokens by probability.
    No safety filtering at all — equivalent to standard top-k sampling.
    
    This is the "NoControl" case from the original paper.
    """

    def __init__(self, top_k: int = 30):
        self.top_k = top_k

    def scan(self, x, P, h_x, lcf, tokenizer) -> FilterResult:
        R = FilterResult()

        # Get the top-k token indices by probability
        topk_values, topk_indices = P.topk(self.top_k)
        R.allowed = topk_indices.cpu().tolist()

        # Optionally compute lcf_mapping for analysis
        # (disabled by default for speed — enable when you need it)

        return R