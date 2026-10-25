# filters/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class FilterResult:
    """
    Result of a filter scan over candidate tokens.
    
    allowed: token indices that passed the filter
    disallowed: token indices that were rejected
    lcf_mapping: maps token_id → h(concat(x, token)) for every evaluated token
    """
    allowed: List[int] = field(default_factory=list)
    disallowed: List[int] = field(default_factory=list)
    lcf_mapping: Dict[int, float] = field(default_factory=dict)


class BaseFilter(ABC):
    """
    Abstract base for all token filters.
    
    Every filter takes:
      - x: current token sequence (1D tensor of token IDs)
      - P: probability distribution over vocab (1D tensor, sums to ~1)
      - h_x: current L-CF value h(x) for the current text
    
    And returns a FilterResult saying which tokens are allowed/disallowed.
    
    NOTE: We pass h_x explicitly rather than recomputing it inside the filter.
    This avoids redundant L-CF calls since the generator already knows h(x)
    from the previous step.
    """
    @abstractmethod
    def scan(
        self,
        x: torch.Tensor,
        P: torch.Tensor,
        h_x: float,
        lcf,  # BaseLanguageCF instance
        tokenizer,  # LLM's tokenizer (for decoding candidate texts)
    ) -> FilterResult:
        pass