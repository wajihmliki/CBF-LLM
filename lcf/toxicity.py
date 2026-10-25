# cbf_llm/lcf/toxicity.py

from typing import List
from .base import BaseLanguageCF


class ToxicityLCF(BaseLanguageCF):
    """
    Toxicity-based L-CF using the Detoxify library.
    
    Detoxify returns a toxicity score in [0, 1], where:
      - 0 means non-toxic
      - 1 means highly toxic
    
    We convert this to our L-CF convention where h(x) >= 0 means safe:
      h(x) = threshold - toxicity_score
    
    So if threshold = 0.5:
      - toxicity_score = 0.1 → h = 0.4  (safe, comfortably)
      - toxicity_score = 0.5 → h = 0.0  (boundary)
      - toxicity_score = 0.9 → h = -0.4 (unsafe)
    
    The threshold controls where we draw the line between S and S_bar.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_type: str = "original",
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        threshold: float
            Toxicity score above this is considered unsafe.
            Default 0.5 is standard. Lower = stricter.
        model_type: str
            Detoxify model variant. Options: "original", "unbiased", "multilingual".
            "original" is fastest and fine for English.
        device: str
            "cuda" or "cpu"
        """
        super().__init__(name=f"toxicity_detoxify_{model_type}")
        self.threshold = threshold

        # Import here so detoxify is only required when this L-CF is used
        from detoxify import Detoxify
        self.model = Detoxify(model_type, device=device)

    def _compute(self, texts: List[str]) -> List[float]:
        """
        For each text, compute h(x) = threshold - toxicity_score.
        
        Detoxify returns multiple scores (toxicity, severe_toxicity, 
        obscene, threat, insult, identity_attack). We use the main 
        'toxicity' score as our primary signal.
        
        Steps:
        1. Run batch through detoxify → dict of score lists
        2. Extract the 'toxicity' scores
        3. Convert: h(x) = threshold - toxicity_score
        """
        results = self.model.predict(texts)
        toxicity_scores = results["toxicity"]

        # Convert to L-CF convention: positive = safe, negative = unsafe
        h_values = [self.threshold - score for score in toxicity_scores]
        return h_values