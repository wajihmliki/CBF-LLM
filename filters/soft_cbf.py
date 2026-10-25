# filters/soft_cbf.py

import torch
import math
from .base import BaseFilter, FilterResult


class SoftCBFFilter(BaseFilter):
    """
    Soft CBF filter: instead of binary allow/reject, applies a 
    smooth weight to each token based on how well it satisfies
    the CBF constraint.
    
    Weight = sigmoid(λ * (h(x⁺) - threshold))
    
    Tokens well above threshold: weight ≈ 1 (unchanged)
    Tokens at threshold: weight = 0.5
    Tokens well below threshold: weight ≈ 0 (effectively removed)
    
    Key advantage over hard filter: preserves more of the LLM's
    original distribution near the boundary, producing more fluent text.
    
    Key advantage over our first soft attempt: uses iterative scanning
    like the hard filter (finds top_k tokens) instead of fixed evaluation.
    """

    def __init__(self, top_k=30, alpha_min=0.1, alpha_max=0.9,
                 beta=5.0, epsilon=0.0, lam=100.0, chunk_size=32):
        """
        Parameters
        ----------
        alpha_min, alpha_max, beta: adaptive α parameters (same as AdaptiveCBF)
        epsilon: safety margin (same as AdaptiveCBF)
        lam: sigmoid sharpness. Higher = closer to hard filter.
             100 = smooth but effective. 1000 = nearly hard.
        """
        self.top_k = top_k
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta = beta
        self.epsilon = epsilon
        self.lam = lam
        self.chunk_size = chunk_size

    def _get_adaptive_alpha(self, h_x):
        sigmoid = 1.0 / (1.0 + math.exp(-self.beta * h_x))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid

    def scan(self, x, P, h_x, lcf, tokenizer) -> FilterResult:
        R = FilterResult()
        sorted_indices = P.argsort(descending=True).cpu().tolist()

        alpha = self._get_adaptive_alpha(h_x)
        threshold = (1.0 - alpha) * h_x + self.epsilon

        # Iterative scan like hard filters — find enough tokens
        penalties = {}
        allowed_count = 0
        idx = 0

        while allowed_count < self.top_k and idx < len(sorted_indices):
            chunk_end = min(idx + self.chunk_size, len(sorted_indices))
            chunk_tokens = sorted_indices[idx:chunk_end]

            candidate_texts = []
            for t in chunk_tokens:
                cand = torch.cat([x, torch.tensor([t], device=x.device)])
                candidate_texts.append(tokenizer.decode(cand))

            h_values = lcf.get_batch(candidate_texts)

            for t, h_next in zip(chunk_tokens, h_values):
                R.lcf_mapping[t] = h_next

                # Sigmoid weight: smooth transition around threshold
                weight = 1.0 / (1.0 + math.exp(-self.lam * (h_next - threshold)))

                penalties[t] = weight

                if weight >= 0.01:  # not effectively zero
                    R.allowed.append(t)
                    allowed_count += 1
                    if allowed_count >= self.top_k:
                        break
                else:
                    R.disallowed.append(t)

            idx = chunk_end

        R.penalties = penalties
        return R