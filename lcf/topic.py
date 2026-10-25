# cbf_llm/lcf/topic.py

import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseLanguageCF


class TopicLCF(BaseLanguageCF):
    """
    Topic adherence L-CF using a Natural Language Inference (NLI) model.
    
    Given a target topic (e.g., "science", "sports"), this L-CF measures
    how related the generated text is to that topic.
    
    Uses an NLI model to check: "Does the text entail the topic?"
    NLI models output 3 classes: contradiction, neutral, entailment.
    
    We construct h(x) = entailment_score - contradiction_score.
    
    h(x) >= 0 means the text is on-topic (safe set S).
    h(x) <  0 means the text contradicts or is off-topic (unsafe set S_bar).
    """

    def __init__(
        self,
        topic: str,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        topic: str
            The target topic, e.g., "This text is about science."
            More specific hypotheses work better than single words.
        model_name: str
            HuggingFace NLI model. Default is BART-large-MNLI, 
            which is the standard zero-shot classification backbone.
        device: str
            "cuda" or "cpu"
        """
        super().__init__(name=f"topic_{topic[:20]}")
        self.topic = topic
        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # BART-large-MNLI label order: contradiction=0, neutral=1, entailment=2
        self.entailment_idx = 2
        self.contradiction_idx = 0

    @torch.no_grad()
    def _compute(self, texts: List[str]) -> List[float]:
        """
        For each text, compute h(x) = entailment_score - contradiction_score.
        
        NLI setup: premise = generated text, hypothesis = topic.
        
        Steps:
        1. Pair each text (premise) with the topic (hypothesis)
        2. Run through NLI model → logits of shape (batch, 3)
        3. Softmax → probabilities
        4. h(x) = P(entailment) - P(contradiction)
        """
        # NLI expects pairs: (premise, hypothesis)
        pairs = [(text, self.topic) for text in texts]
        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        entailment = probs[:, self.entailment_idx]
        contradiction = probs[:, self.contradiction_idx]

        h_values = (entailment - contradiction).cpu().tolist()
        return h_values