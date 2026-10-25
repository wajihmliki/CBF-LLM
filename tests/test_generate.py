
# test_generate.py
# First real end-to-end test. Uses GPT-2 (small, fast) just to verify the pipeline.
# We'll switch to Llama later for real experiments.

from lcf import SentimentLCF
from filters import TopKFilter, CBFFilter, AdaptiveCBFFilter
from generator import CBFGenerator, print_generation_summary

DEVICE = "cuda"  # change to "cpu" if no GPU
PROMPT = "Everyone says you will be a good researcher in the future, but"
MAX_TOKENS = 30

# Use GPT-2 for this test — it's tiny and downloads fast.
# We'll switch to Llama for real experiments.
MODEL = "gpt2"

# Load the L-CF once, share across all generators
print("Loading L-CF...")
lcf = SentimentLCF(device=DEVICE)

# --- Test 1: NoControl (just top-k, no safety) ---
print("\n" + "="*60)
print("TEST 1: NoControl (TopK only)")
print("="*60)
gen_nc = CBFGenerator(MODEL, lcf, TopKFilter(top_k=30), temperature=1.0, device=DEVICE)
result_nc = gen_nc.generate(PROMPT, max_new_tokens=MAX_TOKENS, seed=42)
print_generation_summary(result_nc)

# --- Test 2: Standard CBF (α=0.3) ---
print("\n" + "="*60)
print("TEST 2: CBF Filter (α=0.3)")
print("="*60)
gen_cbf = CBFGenerator(MODEL, lcf, CBFFilter(top_k=30, alpha=0.3), temperature=1.0, device=DEVICE)
result_cbf = gen_cbf.generate(PROMPT, max_new_tokens=MAX_TOKENS, seed=42)
print_generation_summary(result_cbf)

# --- Test 3: Our Adaptive CBF ---
print("\n" + "="*60)
print("TEST 3: Adaptive CBF (our method)")
print("="*60)
gen_acbf = CBFGenerator(MODEL, lcf, AdaptiveCBFFilter(top_k=30, epsilon=0.05), temperature=1.0, device=DEVICE)
result_acbf = gen_acbf.generate(PROMPT, max_new_tokens=MAX_TOKENS, seed=42)
print_generation_summary(result_acbf)

# --- Compare ---
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Filter':<25} {'Disallowed':>12} {'Min h':>10} {'Final h':>10}")
print("-"*60)
for name, r in [("NoControl", result_nc), ("CBF(α=0.3)", result_cbf), ("AdaptiveCBF", result_acbf)]:
    print(f"{name:<25} {r.total_disallowed:>12} {min(r.h_trajectory):>10.4f} {r.h_trajectory[-1]:>10.4f}")