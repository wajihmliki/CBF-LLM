# generator.py

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from filters.base import BaseFilter, FilterResult
from lcf.base import BaseLanguageCF


@dataclass
class GenerationResult:
    """
    Everything we need for analysis from a single generation run.
    
    text: the final generated text (string)
    tokens: list of token IDs generated (not including the prompt)
    h_trajectory: h(x) value at each time step k
    allowed_counts: how many tokens were allowed at each step
    disallowed_counts: how many tokens were disallowed at each step
    total_disallowed: sum of all disallowed tokens across all steps
    """
    text: str = ""
    prompt: str = ""
    tokens: List[int] = field(default_factory=list)
    h_trajectory: List[float] = field(default_factory=list)
    allowed_counts: List[int] = field(default_factory=list)
    disallowed_counts: List[int] = field(default_factory=list)
    total_disallowed: int = 0
    # Store per-step details for deep analysis
    step_details: List[Dict[str, Any]] = field(default_factory=list)


class CBFGenerator:
    """
    Main text generation loop implementing the CBF-LLM pipeline.
    
    Pipeline at each step k:
      1. LLM takes current text x(k) → outputs logits
      2. Token predictor converts logits → probability P (softmax with temperature)
      3. Filter scans P using L-CF → returns allowed/disallowed tokens
      4. Normalizer adjusts P to zero out disallowed, renormalize → Q
      5. Token selector samples from Q → next token t*
      6. Concatenate: x(k+1) = concat(x(k), t*)
    
    This matches Figure 3 in the paper exactly.
    """

    def __init__(
        self,
        model_name: str,
        lcf: BaseLanguageCF,
        token_filter: BaseFilter,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        model_name: str
            HuggingFace model name for the generative LLM.
            e.g., "meta-llama/Llama-3.2-1B" or "gpt2" for quick testing.
        lcf: BaseLanguageCF
            The language-constraint function h(x).
        token_filter: BaseFilter
            The filter F (TopK, Blacklist, CBF, or AdaptiveCBF).
        temperature: float
            Softmax temperature. Higher = more random, lower = more focused.
        device: str
            "cuda" or "cpu"
        """
        self.device = device
        self.temperature = temperature
        self.lcf = lcf
        self.token_filter = token_filter

        # Load the generative LLM and its tokenizer
        print(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # half precision to save GPU memory
        ).to(device)
        self.model.eval()
        print(f"LLM loaded on {device}.")

        # Set pad token if not present (common for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt using the CBF-LLM pipeline.
        
        Parameters
        ----------
        prompt: str
            Initial text x(0).
        max_new_tokens: int
            Maximum number of tokens to generate.
        seed: Optional[int]
            Random seed for reproducibility. None = no seed.
        
        Returns
        -------
        GenerationResult with full trajectory data for analysis.
        """
        if seed is not None:
            torch.manual_seed(seed)

        result = GenerationResult(prompt=prompt)

        # Step 0: Encode the prompt into token IDs
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.device)
        # input_ids is now a 1D tensor: [t1, t2, ..., tn]

        # Compute initial h(x0)
        h_current = self.lcf.get(prompt)
        result.h_trajectory.append(h_current)

        # Generation loop: k = 0, 1, 2, ...
        for k in range(max_new_tokens):
            # --- Step 1: Token Predictor G ---
            # Feed current token sequence into the LLM
            # Output: logits of shape (sequence_length, vocab_size)
            # We only care about the LAST position (next token prediction)
            outputs = self.model(input_ids.unsqueeze(0))  # add batch dim
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # --- Step 2: Convert logits to probability P ---
            # P = softmax(logits / temperature)
            # This is eq (8) from the paper
            P = torch.softmax(logits / self.temperature, dim=0)

            # --- Step 3: Filter scans P ---
            # The filter decides which tokens are allowed based on the L-CF
            filter_result = self.token_filter.scan(
                x=input_ids,
                P=P,
                h_x=h_current,
                lcf=self.lcf,
                tokenizer=self.tokenizer,
            )

            # --- Step 4: Normalizer ---
            # Zero out disallowed tokens, renormalize so Q sums to 1
            # This is eq (12) from the paper
            Q = torch.zeros_like(P)
            if len(filter_result.allowed) > 0:
                allowed_tensor = torch.tensor(filter_result.allowed, device=self.device)
                Q[allowed_tensor] = P[allowed_tensor]

                # Apply soft penalties if available
                if hasattr(filter_result, 'penalties') and filter_result.penalties:
                    for t, weight in filter_result.penalties.items():
                        if Q[t] > 0:
                            Q[t] *= weight

                q_sum = Q.sum()
                if q_sum > 0 and not torch.isnan(q_sum) and not torch.isinf(q_sum):
                    Q = Q / q_sum
                else:
                    Q = torch.zeros_like(P)
                    Q[P.argmax()] = 1.0
            else:
                Q = torch.zeros_like(P)
                Q[P.argmax()] = 1.0
                print(f"  WARNING at step {k}: No tokens allowed, falling back to argmax")

            # --- Step 5: Token Selector C ---
            # Sample next token from Q
            next_token = torch.multinomial(Q, num_samples=1).item()

            # --- Step 6: Concatenate ---
            # x(k+1) = concat(x(k), t*)
            input_ids = torch.cat([input_ids, torch.tensor([next_token], device=self.device)])

            # Compute h(x(k+1)) for the new text
            new_text = self.tokenizer.decode(input_ids)
            h_current = self.lcf.get(new_text)

            # Record everything for analysis
            result.tokens.append(next_token)
            result.h_trajectory.append(h_current)
            result.allowed_counts.append(len(filter_result.allowed))
            result.disallowed_counts.append(len(filter_result.disallowed))
            result.total_disallowed += len(filter_result.disallowed)
            result.step_details.append({
                "k": k,
                "token_id": next_token,
                "token_str": self.tokenizer.decode([next_token]),
                "h_value": h_current,
                "allowed": len(filter_result.allowed),
                "disallowed": len(filter_result.disallowed),
            })

        # Final text
        result.text = self.tokenizer.decode(input_ids)
        return result


def print_generation_summary(result: GenerationResult):
    """Pretty-print a generation result for quick inspection."""
    print(f"\nPrompt: {result.prompt}")
    print(f"Generated: {result.text}")
    print(f"Total disallowed tokens: {result.total_disallowed}")
    print(f"h trajectory: {[f'{h:.3f}' for h in result.h_trajectory]}")
    print(f"Final h(x): {result.h_trajectory[-1]:.4f}")
    h_min = min(result.h_trajectory)
    print(f"Min h(x) during generation: {h_min:.4f}")
    if h_min < 0:
        print("  ⚠ Text entered unsafe region during generation!")
    else:
        print("  ✓ Text stayed in safe region throughout.")

class SharedModelGenerator:
    """
    Like CBFGenerator but lets you share a single loaded LLM across
    multiple experiments. Avoids reloading the model hundreds of times.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        print(f"Loading shared LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Shared LLM loaded on {device}.")

    def generate(
        self,
        prompt: str,
        lcf: BaseLanguageCF,
        token_filter: BaseFilter,
        temperature: float = 1.0,
        max_new_tokens: int = 30,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)

        result = GenerationResult(prompt=prompt)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.device)
        h_current = lcf.get(prompt)
        result.h_trajectory.append(h_current)

        for k in range(max_new_tokens):
            outputs = self.model(input_ids.unsqueeze(0))
            logits = outputs.logits[0, -1, :]
            P = torch.softmax(logits / temperature, dim=0)

            filter_result = token_filter.scan(
                x=input_ids, P=P, h_x=h_current,
                lcf=lcf, tokenizer=self.tokenizer,
            )

            Q = torch.zeros_like(P)
            if len(filter_result.allowed) > 0:
                allowed_tensor = torch.tensor(filter_result.allowed, device=self.device)
                Q[allowed_tensor] = P[allowed_tensor]

                # Apply soft penalties if available
                if hasattr(filter_result, 'penalties') and filter_result.penalties:
                    for t, weight in filter_result.penalties.items():
                        if Q[t] > 0:
                            Q[t] *= weight

                q_sum = Q.sum()
                if q_sum > 0 and not torch.isnan(q_sum) and not torch.isinf(q_sum):
                    Q = Q / q_sum
                else:
                    Q = torch.zeros_like(P)
                    Q[P.argmax()] = 1.0
            else:
                Q = torch.zeros_like(P)
                Q[P.argmax()] = 1.0

            next_token = torch.multinomial(Q, num_samples=1).item()
            input_ids = torch.cat([input_ids, torch.tensor([next_token], device=self.device)])
            new_text = self.tokenizer.decode(input_ids)
            h_current = lcf.get(new_text)

            result.tokens.append(next_token)
            result.h_trajectory.append(h_current)
            result.allowed_counts.append(len(filter_result.allowed))
            result.disallowed_counts.append(len(filter_result.disallowed))
            result.total_disallowed += len(filter_result.disallowed)

        result.text = self.tokenizer.decode(input_ids)
        return result