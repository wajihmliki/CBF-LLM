# cbf_llm/lcf/sentiment.py

import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseLanguageCF


class SentimentLCF(BaseLanguageCF):
    """
    Sentiment-based L-CF using a RoBERTa sentiment classifier.
    
    Implements eq (13) from the paper:
        s = softmax(M(x))
        h(x) = s[positive] - max(s[negative], s[neutral])
    
    h(x) >= 0 when text is classified as positive sentiment.
    h(x) <  0 when text is negative or neutral.
    
    The original paper uses 'cardiffnlp/twitter-roberta-base-sentiment-latest'.
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "cuda",
        positive_index: int = 2,  # Index of the "positive" class in the model output
    ):
        super().__init__(name=f"sentiment_{model_name.split('/')[-1]}")
        self.device = device
        self.positive_index = positive_index

        # Load the classifier model and its tokenizer.
        # IMPORTANT: this is a SEPARATE model from the generative LLM.
        # The L-CF model is typically small (RoBERTa-base = ~125M params).
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def _compute(self, texts: List[str]) -> List[float]:
        """
        For each text, compute h(x) = s[positive] - max(s[other classes]).
        
        Steps:
        1. Tokenize the batch of texts
        2. Run through the RoBERTa classifier → logits of shape (batch, 3)
        3. Apply softmax → probabilities s of shape (batch, 3)
        4. h(x) = s[positive_index] - max over other indices
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).logits  # (batch_size, num_classes)
        probs = torch.softmax(logits, dim=-1)  # (batch_size, num_classes)

        # s[positive]
        positive_scores = probs[:, self.positive_index]  # (batch_size,)

        # max(s[other classes]): mask out the positive index, take max
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[:, self.positive_index] = False
        other_max = probs.masked_fill(~mask, -float('inf')).max(dim=-1).values

        h_values = positive_scores - other_max  # (batch_size,)
        return h_values.cpu().tolist()