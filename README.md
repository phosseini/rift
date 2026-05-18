# RIFT — RubrIc Failure mode Taxonomy

Automated diagnostics for rubric quality. RIFT classifies rubric criteria against a taxonomy of eight failure modes organized into three categories: **Reliability**, **Content Validity**, and **Consequential Validity**.

> **Paper:** [RIFT: A RubrIc Failure Mode Taxonomy and Automated Diagnostics](https://arxiv.org/abs/2604.01375)


## Installation

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo>
cd rift
uv sync
cp .env.example .env   # add your API keys
```

`.env` keys:
```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```


## Quick smoke test

```bash
# Prevalence experiment — 3 rubrics per source, fast sanity check
uv run python prevalence_experiment.py --n 3 --concurrency 5 --judge gpt-5.4-2026-03-05

# HBP experiment — 3 conversations
uv run python hbp_experiment.py --n 3 --concurrency 3 --judge gpt-5.4-2026-03-05 --eval-strategy scoped
```


## Experiments

### 1. Prevalence experiment (`prevalence_experiment.py`)

Reproduces RIFT paper Table 2. Evaluates failure mode prevalence across five rubric datasets using the **joined** strategy (full rubric evaluated with all failure modes — paper-equivalent method).

```bash
uv run python prevalence_experiment.py --n 50 --concurrency 10 --judge gpt-5.2-2025-12-11
```

Results are saved to `results/prevalence_<timestamp>.jsonl` and cached per judge. Each record includes `rubric_text`, `labels` (majority-voted), `votes` (raw per-run outputs), `n_votes`, and an `error` field if the API call failed.

| Parameter | Default | Description |
|---|---|---|
| `--n` | `10` | Number of rubrics per source (5 sources → n×5 total API calls) |
| `--concurrency` | `10` | Max simultaneous API calls |
| `--judge` | `gpt-5.4-2026-03-05 gemini-3.1-pro-preview` | One or more judge models (space-separated) |
| `--votes` | `1` | Number of judge runs per rubric; majority vote when >1 (paper uses 5) |
| `--no-cache` | off | Force re-run even if cached results exist |


### 2. HBP experiment (`hbp_experiment.py`)

Runs RIFT on all 525 conversations and 1,135 rubric criteria from [HealthBench Professional](https://huggingface.co/datasets/openai/healthbench-professional).

```bash
uv run python hbp_experiment.py --concurrency 8 --judge gpt-5.4-2026-03-05 --eval-strategy scoped
```

Results are saved to `results/hbp_<timestamp>.jsonl` and cached per judge + strategy. Each record includes `rubric_text`, `labels` (majority-voted), `votes` (raw per-run outputs), `n_votes`, and an `error` field if the API call failed.

| Parameter | Default | Description |
|---|---|---|
| `--eval-strategy` | `scoped` | Evaluation strategy (see table below) |
| `--concurrency` | `2` | Max simultaneous API calls |
| `--judge` | `gpt-5.4-2026-03-05` | One or more judge models (space-separated) |
| `--n` | all 525 | Limit to first N conversations |
| `--votes` | `1` | Number of judge runs per rubric; majority vote when >1 (paper uses 5) |
| `--no-cache` | off | Force re-run even if cached results exist |

#### Evaluation strategies

| Strategy | Description | eval_mode in results |
|---|---|---|
| `joined` | All rubric criteria for a conversation concatenated into one string, evaluated with all enabled failure modes. Equivalent to the paper's method applied at conversation level. | `joined` |
| `scoped` | Criterion-scope failure modes evaluated on each criterion individually; rubric-scope modes evaluated on the full joined rubric. Avoids inflating rubric-level failure modes on single criteria. | `per_criterion` + `per_conversation` |


## Judges

| Model ID | Provider | Notes |
|---|---|---|
| `gpt-5.2-2025-12-11` | OpenAI | Paper's primary judge |
| `gpt-5.4-2026-03-05` | OpenAI | Latest OpenAI judge |
| `gemini-3.1-pro-preview` | Google | Latest Gemini judge |

Pass one or more judges via `--judge`. Results for each judge are cached and analyzed separately.


## Failure modes

| Mode | Scope | Category | Description |
|---|---|---|---|
| `subjective` | criterion | Reliability | Uses unanchored subjective terms |
| `non_atomic` | criterion | Reliability | Bundles multiple independently scorable requirements |
| `ungrounded` | criterion | Reliability | Requires verification without providing grounding |
| `misaligned_or_rigid` | criterion | Content Validity | Grades wrong objective or over-constrains |
| `missing_criteria` | rubric | Content Validity | Prompt implies requirements the rubric doesn't cover |
| `hackable` | criterion | Consequential Validity | Gameable via proxy metrics |
| `low_signal` | rubric | Consequential Validity | Rubric as a whole doesn't discriminate well |
| `redundant_criteria` | rubric | Consequential Validity | Multiple criteria evaluate the same requirement |

**Scope** determines which evaluation level is used in `scoped` strategy: `criterion` modes run on individual criteria; `rubric` modes run on the full joined rubric per conversation.

Scopes can be overridden per-experiment in `config.json`.


## Configuration (`config.json`)

Controls which failure modes are enabled and their scopes. Remove a mode to exclude it from all experiments. Change a scope value to override the default.

```json
{
  "failure_modes": {
    "subjective":          "criterion",
    "non_atomic":          "criterion",
    "ungrounded":          "criterion",
    "misaligned_or_rigid": "criterion",
    "missing_criteria":    "rubric",
    "hackable":            "criterion",
    "low_signal":          "rubric",
    "redundant_criteria":  "rubric"
  }
}
```


## Reference

```bibtex
@article{qi2025rift,
  title   = {RIFT: A RubrIc Failure Mode Taxonomy and Automated Diagnostics},
  author  = {Qi, Zhengyang and Dickens, Charles and Pham, Derek and Dsouza, Amanda and
             Parchami, Armin and Sala, Frederic and Varma, Paroma},
  journal = {arXiv preprint arXiv:2604.01375},
  year    = {2025}
}
```
