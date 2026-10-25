# AdaptiveCBF: Robust Safe Decoding for LLMs via Adaptive Control Barrier Functions

Official implementation of **"Robust Safe Decoding for Large Language Models via Adaptive Control Barrier Functions"** (paper in [`paper/`](paper/)).

CBF-based decoding ([CBF-LLM](https://arxiv.org/abs/2408.15625)) filters candidate tokens at each generation step using a safety classifier `h(x)`, trusting it as ground truth. In practice that classifier is noisy. This repo implements **AdaptiveCBF**, a drop-in, training-free replacement that adds:

- an **adaptive strictness parameter** `α(h_x)` — strict near the safety boundary, relaxed when deep in the safe region,
- an explicit **safety margin `ε`** that compensates for classifier noise,
- **AdaptiveCBF-BO**, a boundary-only variant of the margin for efficiency,
- **SmoothCBF**, a soft-weighting relaxation that trades worst-case guarantees for fluency.

See the paper for the probabilistic safety guarantee `P(safe) ≥ Φ(ε/σ)^N` and the full experimental results.

## Repo structure

```
filters/        Token filters: NoControl (top-k), Blacklist, CBF, AdaptiveCBF(-BO), SmoothCBF
lcf/             Language-Constraint Functions h(x): toxicity (Detoxify), sentiment (RoBERTa), topic (BART-MNLI)
generator.py      CBF decoding loop (LLM logits -> filter -> renormalize -> sample)
experiment.py      Main runner: sweeps noise levels x filters, used for Tables 1, 3, 4, 5, 6
experiment_swap.py  Classifier-mismatch runner, used for Table 7
tests/            Minimal smoke tests (imports, filter math, end-to-end generation on GPT-2)
results/          Raw JSON outputs behind Tables 1, 3, 5, 6 (Qwen2.5-1.5B)
paper/            The paper PDF
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `meta-llama/Llama-3.2-1B` is a gated model on HuggingFace — you'll need to accept the license on the model page and run `huggingface-cli login` before using `--model meta-llama/Llama-3.2-1B`.
- `detoxify`, the sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`), and the topic model (`facebook/bart-large-mnli`) download automatically on first use.
- A CUDA GPU is strongly recommended; pass `--device cpu` otherwise (slow).

## Quick sanity check

```bash
python -m tests.test_setup     # checks imports + L-CF scores load correctly
python -m tests.test_filters    # checks adaptive-alpha math
python -m tests.test_generate    # end-to-end generation on GPT-2 (fast, no gated models)
```

## Reproducing the paper's results

All runs use top-k=30 sampling, temperature=1.0, 30 new tokens, and `ε = 1.5σ` (per Corollary 2, giving a per-step safety probability of `Φ(1.5) ≈ 0.933`). `experiment.py` sweeps `σ ∈ {0.0, 0.1, 0.2, 0.3, 0.4}` and all six filters automatically within one run.

**Table 1 (Toxicity, RTP, Qwen2.5-1.5B)** — also the source of `results/paper_rtp_final.json`:
```bash
python experiment.py --task toxicity --source rtp --model Qwen/Qwen2.5-1.5B \
  --num_prompts 100 --num_seeds 20 --max_tokens 30 --output results/paper_rtp_final.json
```

**Table 3 (Toxicity, BeaverTails, Qwen2.5-1.5B)** — `results/paper_beaver_final.json`:
```bash
python experiment.py --task toxicity --source beaver --model Qwen/Qwen2.5-1.5B \
  --num_prompts 100 --num_seeds 20 --max_tokens 30 --output results/paper_beaver_final.json
```

**Table 4 (Cross-model, Llama-3.2-1B columns)** — rerun the RTP/BeaverTails commands above with `--model meta-llama/Llama-3.2-1B`. *(Not included under `results/` — regenerate to reproduce.)*

**Table 5 (Sentiment control)** — `results/paper_sentiment_final.json`:
```bash
python experiment.py --task sentiment --source rtp --model Qwen/Qwen2.5-1.5B \
  --num_prompts 100 --num_seeds 20 --max_tokens 30 --output results/paper_sentiment_final.json
```

**Table 6 (Topic adherence)** — `results/paper_topic_final.json`:
```bash
python experiment.py --task topic --source rtp --model Qwen/Qwen2.5-1.5B \
  --num_prompts 100 --num_seeds 20 --max_tokens 30 --output results/paper_topic_final.json
```

**Table 7 (Classifier swap / mismatch)** — *(not included under `results/`; regenerate)*:
```bash
python experiment_swap.py --model Qwen/Qwen2.5-1.5B --num_prompts 50 --num_seeds 20
```
This filters generation with Detoxify at threshold 0.15 and evaluates true safety with the same classifier at threshold 0.10, using `ε = 0.05` (the threshold gap) for AdaptiveCBF.

> **Note:** this repo currently ships the raw result JSONs behind Tables 1, 3, 5, 6 only. The Llama columns of Table 4, Table 7, and the §6.6 perplexity/diversity numbers are reproducible via the commands above but their raw outputs weren't archived — PRs welcome.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{imliki2026adaptivecbf,
  title={Robust Safe Decoding for Large Language Models via Adaptive Control Barrier Functions},
  author={Imliki, Wajih and Deresa, Chala Adane and Choi, Han-Lim},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
