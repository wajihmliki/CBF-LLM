# filters/cbf_filter.py

import torch
from .base import BaseFilter, FilterResult


class CBFFilter(BaseFilter):
    """
    CBF filter: allows token t only if it satisfies the discrete-time
    CBF constraint:
    
        h(concat(x, t)) - h(x) >= -α * h(x)
    
    Which rearranges to:
    
        h(concat(x, t)) >= (1 - α) * h(x)
    
    This is the core method from the original CBF-LLM paper (Algorithm 2).
    
    α ∈ [0, 1] controls strictness:
      - α = 0: strictest. Only allows tokens where h increases or stays same.
      - α = 1: mildest. Equivalent to blacklist (just requires h >= 0).
      - α = 0.3 or 0.8: intermediate values tested in the paper.
    """

    def __init__(self, top_k: int = 30, alpha: float = 0.3, chunk_size: int = 32):
        self.top_k = top_k
        self.alpha = alpha
        self.chunk_size = chunk_size
        # Precompute (1 - α) since it's used every scan
        self.one_minus_alpha = 1.0 - alpha

    def scan(self, x, P, h_x, lcf, tokenizer) -> FilterResult:
        R = FilterResult()
        sorted_indices = P.argsort(descending=True).cpu().tolist()

        # The CBF threshold: h(x⁺) must be >= this value
        threshold = self.one_minus_alpha * h_x

        allowed_count = 0
        idx = 0

        while allowed_count < self.top_k and idx < len(sorted_indices):
            chunk_end = min(idx + self.chunk_size, len(sorted_indices))
            chunk_tokens = sorted_indices[idx:chunk_end]

            candidate_texts = []
            for t in chunk_tokens:
                candidate_ids = torch.cat([x, torch.tensor([t], device=x.device)])
                candidate_texts.append(tokenizer.decode(candidate_ids))

            h_values = lcf.get_batch(candidate_texts)

            for t, h_next in zip(chunk_tokens, h_values):
                R.lcf_mapping[t] = h_next

                if h_next >= threshold:
                    R.allowed.append(t)
                    allowed_count += 1
                    if allowed_count >= self.top_k:
                        break
                else:
                    R.disallowed.append(t)

            idx = chunk_end

        return R