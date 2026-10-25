# filters/blacklist.py

import torch
from .base import BaseFilter, FilterResult


class BlacklistFilter(BaseFilter):
    """
    Blacklist filter: allows a token only if h(concat(x, t)) >= 0.
    
    This is equivalent to the CBF filter with α = 1.
    It's the simplest safety filter — just block any token that would
    make the text cross into the unsafe set S_bar.
    
    The original paper includes this as a baseline comparison.
    """

    def __init__(self, top_k: int = 30, chunk_size: int = 32):
        self.top_k = top_k
        self.chunk_size = chunk_size

    def scan(self, x, P, h_x, lcf, tokenizer) -> FilterResult:
        R = FilterResult()
        sorted_indices = P.argsort(descending=True).cpu().tolist()

        allowed_count = 0
        idx = 0

        while allowed_count < self.top_k and idx < len(sorted_indices):
            # Process in chunks for efficient batched L-CF evaluation
            chunk_end = min(idx + self.chunk_size, len(sorted_indices))
            chunk_tokens = sorted_indices[idx:chunk_end]

            # Build candidate texts: concat(x, t) for each token t in chunk
            candidate_texts = []
            for t in chunk_tokens:
                candidate_ids = torch.cat([x, torch.tensor([t], device=x.device)])
                candidate_texts.append(tokenizer.decode(candidate_ids))

            # Batch evaluate h(concat(x, t)) for all candidates
            h_values = lcf.get_batch(candidate_texts)

            for t, h_next in zip(chunk_tokens, h_values):
                R.lcf_mapping[t] = h_next

                if h_next >= 0:
                    R.allowed.append(t)
                    allowed_count += 1
                    if allowed_count >= self.top_k:
                        break
                else:
                    R.disallowed.append(t)

            idx = chunk_end

        return R