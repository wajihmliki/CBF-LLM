# cbf_llm/lcf/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class BaseLanguageCF(ABC):
    """
    Abstract base for Language-Constraint Functions h(x).
    
    The L-CF maps text to a real number where:
      h(x) >= 0  means x is in the safe/desirable set S
      h(x) <  0  means x is in the unsafe/undesirable set S_bar
    
    This follows Definition (9) in the CBF-LLM paper.
    """

    def __init__(self, name: str = "unnamed_lcf"):
        self.name = name
        # Cache to avoid redundant model calls.
        # Key: string text, Value: h(x) float
        self._cache: Dict[str, float] = {}

    @abstractmethod
    def _compute(self, texts: List[str]) -> List[float]:
        """
        Core computation: given a batch of text strings,
        return h(x) for each. Subclasses implement this.
        """
        pass

    def get(self, text: str) -> float:
        """Get h(x) for a single text string."""
        return self.get_batch([text])[0]

    def get_batch(self, texts: List[str]) -> List[float]:
        """
        Get h(x) for a batch of texts, using cache where possible.
        
        This is the main entry point. It checks the cache first,
        computes only for uncached texts, then merges results.
        """
        # Split into cached and uncached
        uncached_texts = []
        uncached_indices = []
        results = [0.0] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Compute uncached ones in a batch
        if uncached_texts:
            computed = self._compute(uncached_texts)
            for idx, text, value in zip(uncached_indices, uncached_texts, computed):
                self._cache[text] = value
                results[idx] = value

        return results

    def clear_cache(self):
        """Clear the cache. Useful between experiments."""
        self._cache.clear()

    def __call__(self, text):
        if isinstance(text, str):
            return self.get(text)
        if isinstance(text, list):
            return self.get_batch(text)
        raise TypeError(f"Expected str or List[str], got {type(text)}")