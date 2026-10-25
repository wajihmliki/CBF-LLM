# filters/adaptive_cbf.py

import torch
from .base import BaseFilter, FilterResult


class AdaptiveCBFFilter(BaseFilter):
    """
    Adaptive CBF filter with safety margin for evaluator uncertainty.
    
    THIS IS OUR CONTRIBUTION — not in the original paper.
    
    Two key extensions over the standard CBF filter:
    
    1. Safety margin ε: Instead of requiring h(x⁺) >= (1-α)*h(x),
       we require h(x⁺) >= (1-α)*h(x) + ε.
       This accounts for the fact that the L-CF classifier might be wrong.
       The margin ε keeps us further from the unsafe boundary.
    
    2. Adaptive α: Instead of fixed α, we adjust α based on the current 
       safety score h(x). When h(x) is large (far from unsafe), α is 
       large (relaxed constraint, fewer interventions). When h(x) is 
       near 0 (close to boundary), α is small (strict constraint, 
       more cautious).
    
    The adaptive formula:
        α_adaptive = α_min + (α_max - α_min) * sigmoid(β * h(x))
    
    Where β controls how sharply α transitions. When h(x) >> 0, 
    sigmoid ≈ 1 so α ≈ α_max (relaxed). When h(x) ≈ 0, 
    sigmoid ≈ 0.5 so α is in the middle. When h(x) << 0, 
    sigmoid ≈ 0 so α ≈ α_min (strict, trying hard to recover).
    """

    def __init__(
        self,
        top_k: int = 30,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        beta: float = 5.0,
        epsilon: float = 0.05,
        chunk_size: int = 32,
        boundary_only: bool = False,

    ):
        """
        Parameters
        ----------
        top_k: int
            Number of tokens to allow through the filter.
        alpha_min: float
            Strictest α value (used when h(x) is low / near boundary).
        alpha_max: float
            Most relaxed α value (used when h(x) is high / safely away).
        beta: float
            Controls how sharply α adapts to h(x). Higher = sharper transition.
        epsilon: float
            Safety margin. Higher = more conservative.
            This is the key robustness parameter.
        chunk_size: int
            Batch size for L-CF evaluations.
        """
        self.top_k = top_k
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta = beta
        self.epsilon = epsilon
        self.chunk_size = chunk_size
        self.boundary_only = boundary_only

    def _get_adaptive_alpha(self, h_x: float) -> float:
        """
        Compute α based on current safety score.
        Uses sigmoid to smoothly transition between α_min and α_max.
        """
        import math
        sigmoid = 1.0 / (1.0 + math.exp(-self.beta * h_x))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid

    def scan(self, x, P, h_x, lcf, tokenizer) -> FilterResult:
        R = FilterResult()
        sorted_indices = P.argsort(descending=True).cpu().tolist()

        # Compute adaptive α based on current h(x)
        alpha = self._get_adaptive_alpha(h_x)

        # The robust CBF threshold: h(x⁺) must be >= this value
        # Standard CBF:  threshold = (1 - α) * h(x)
        # Ours:          threshold = (1 - α) * h(x) + ε
        #threshold = (1.0 - alpha) * h_x + self.epsilon
        #NEW THRESHOLD: only apply margin when near boundary
        # Compute threshold
        if self.boundary_only and self.epsilon > 0:
            effective_eps = self.epsilon * max(0, 1.0 - h_x / (2 * self.epsilon))
        else:
            effective_eps = self.epsilon
        threshold = (1.0 - alpha) * h_x + effective_eps

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