# Abliteration Lab

What does removing an LLM's refusals actually cost? This repo removes refusal behaviour from
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) two ways and measures the damage
with standard benchmarks.

- **From scratch** — `abliterate.py` reproduces classic abliteration
  ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)): find the refusal direction, project it out of the weights.
- **With [Heretic](https://github.com/p-e-w/heretic)** — automatic optimisation using Arbitrary-Rank Ablation (ARA),
  trading off refusals against drift from the original model.

The resulting model is published at
**[aasis001/Qwen2.5-1.5B-Instruct-heretic-ara](https://huggingface.co/aasis001/Qwen2.5-1.5B-Instruct-heretic-ara)**.

> ⚠️ Abliterated models answer requests the original refuses. Use them for research, red-teaming and studying
> refusal mechanisms — not for customer-facing or unsupervised use.

## Results

Refusals: 100 held-out harmful prompts (`mlabonne/harmful_behaviors`), keyword-matched.
KL divergence: drift from the original on harmless prompts (`mlabonne/harmless_alpaca`).
Benchmarks: LM Evaluation Harness 0.4.13, bf16; GSM8K is 5-shot on the first 250 test questions.

| Model | Refusals | KL | MMLU | ARC-C | HellaSwag | WinoGrande | PIQA | GSM8K | GSM8K strict |
|---|---|---|---|---|---|---|---|---|---|
| Original | 99 | 0 | 60.2 | 46.4 | 68.3 | 62.5 | 75.9 | 50.8 | 27.6 |
| **Published model** | **7** | **0.014** | **59.6** | – | – | – | – | **48.4** | **30.4** |
| Alternative A | 6 | 0.039 | 58.8 | 46.2 | 67.2 | 62.7 | 75.7 | 42.8 | 17.2 |
| Alternative B | 0 | 1.477 | 55.0 | 34.3 | 56.4 | 59.1 | 69.5 | 5.2 | 0.0 |

Full numbers with standard errors: [`results/benchmarks.json`](results/benchmarks.json).

**Findings**

- **KL divergence predicts capability damage.** The lower the drift, the smaller the benchmark loss.
- **The last refusals are the expensive ones.** Going from 7 to 0 refusals cost 33–43 points on GSM8K.
- **A low-drift configuration is essentially free.** The published model is within standard error of the
  original on MMLU and GSM8K.
- **Maths is the canary.** Multi-step reasoning (GSM8K) degraded first and most; multiple-choice benchmarks barely moved.

The published model was not re-run on ARC-C, HellaSwag, WinoGrande and PIQA: alternative A, which drifts more,
showed no real loss on them.

## Reproduce

Needs an NVIDIA GPU with 6 GB+ VRAM.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Classic abliteration from scratch** (~10 min):

```bash
python abliterate.py                 # auto-selects the layer
python abliterate.py --layer 16      # or force one
```

**Heretic** (~50 min for 100 trials):

```bash
heretic --model Qwen/Qwen2.5-1.5B-Instruct --seed 42
```

Heretic is non-deterministic across hardware, so your trials will differ. Pick a trial with low KL
(< 0.02) and few refusals from the Pareto front it shows. The published model used these ARA parameters:

| Parameter | Value |
|---|---|
| start_layer_index | 11 |
| end_layer_index | 28 |
| preserve_good_behavior_weight | 0.2704 |
| steer_bad_behavior_weight | 0.0002 |
| overcorrect_relative_weight | 0.0917 |
| neighbor_count | 7 |

**Benchmark** (resumable; ~45 min per model on an RTX 3060):

```bash
./bench.sh original Qwen/Qwen2.5-1.5B-Instruct
./bench.sh heretic  aasis001/Qwen2.5-1.5B-Instruct-heretic-ara
```

**Chat** with either model:

```bash
python chat.py             # published model
python chat.py original    # for comparison
```

Gotcha when re-exporting a trial from a finished Heretic run: `--checkpoint-action continue` restores the settings
stored in the checkpoint, so a new `--trial-index` on the command line is silently ignored.

## Credits

- Base model: Qwen team, Qwen2.5-1.5B-Instruct (Apache 2.0).
- [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann (AGPL-3.0) — used as a tool, no code copied.
- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.
- Datasets: `mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`.
