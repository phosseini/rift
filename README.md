# RIFT — RubrIc Failure mode Taxonomy

Automated diagnostics for rubric quality. RIFT classifies rubric criteria against a taxonomy of eight failure modes organized into three categories: **Reliability**, **Content Validity**, and **Consequential Validity**.

> **Paper:** [RIFT: A RubrIc Failure Mode Taxonomy and Automated Diagnostics](https://arxiv.org/abs/2604.01375)


## Failure modes

| Mode | Category | Description |
|---|---|---|
| `subjective` | Reliability | Uses unanchored subjective terms |
| `non_atomic` | Reliability | Bundles multiple independently scorable requirements |
| `ungrounded` | Reliability | Requires verification without providing grounding |
| `misaligned_or_rigid` | Content Validity | Grades wrong objective or over-constrains |
| `missing_criteria` | Content Validity | Prompt implies requirements the rubric doesn't cover |
| `hackable` | Consequential Validity | Gameable via proxy metrics |
| `low_signal` | Consequential Validity | Rubric as a whole doesn't discriminate well |
| `redundant_criteria` | Consequential Validity | Multiple criteria evaluate the same requirement |


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

Default for `--n` is `10` (5 sources → 50 total API calls). The paper uses `--votes 5`.


### 2. HBP experiment (`hbp_experiment.py`)

Runs RIFT on all 525 conversations and 1,135 rubric criteria from [HealthBench Professional](https://huggingface.co/datasets/openai/healthbench-professional).

```bash
uv run python hbp_experiment.py --concurrency 8 --judge gpt-5.4-2026-03-05 --eval-strategy scoped
```

Results are saved to `results/hbp_<timestamp>.jsonl` and cached per judge + strategy. Each record includes `rubric_text`, `labels` (majority-voted), `votes` (raw per-run outputs), `n_votes`, and an `error` field if the API call failed.

`--eval-strategy` controls how failure modes are applied:

- **`joined`** — all rubric criteria for a conversation are concatenated into one string and evaluated with all failure modes together. Equivalent to the paper's method.
- **`scoped`** (default) — criterion-scope failure modes run on each criterion individually; rubric-scope modes run on the full joined rubric. Produces `per_criterion` and `per_conversation` records, letting you pinpoint failure modes at the criterion level rather than just the rubric.


## Parameters

All experiments share the same CLI parameters:

```
--judge            gpt-5.4-2026-03-05        Judge model(s) to use, space-separated. See Judges section for available models.
--concurrency      10                         Max simultaneous API calls. Lower this if you hit rate limits.
--n                (experiment-specific)      Limit to first N items (rubrics or conversations). Useful for quick tests.
--votes            1                          Number of judge runs per rubric; majority vote is used when >1. The paper uses 5.
--no-cache         off                        Force re-run even if cached results exist for this judge + strategy.
--eval-strategy    scoped                     (HBP only) joined or scoped — see HBP experiment section for details.
```

Defaults for `--n` and `--concurrency` differ per experiment — see each experiment section above.


## Datasets

| Dataset | HuggingFace | Type | Used in |
|---|---|---|---|
| AdvancedIF | 🤗 [facebook/AdvancedIF](https://huggingface.co/datasets/facebook/AdvancedIF) | Human-curated | Prevalence experiment |
| ResearchRubrics | 🤗 [ScaleAI/researchrubrics](https://huggingface.co/datasets/ScaleAI/researchrubrics) | Human-written | Prevalence experiment |
| WildChecklists | 🤗 [viswavi/wildchecklists](https://huggingface.co/datasets/viswavi/wildchecklists) | LLM-generated | Prevalence experiment |
| OpenRubrics | 🤗 [OpenRubrics/OpenRubrics](https://huggingface.co/datasets/OpenRubrics/OpenRubrics) | LLM-generated | Prevalence experiment |
| Auto-Rubric | 🤗 [agentscope-ai/Auto-Rubric](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric) | LLM-generated | Prevalence experiment |
| HealthBench Professional | 🤗 [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) | Physician-written | HBP experiment |


## Judges

| Model ID | Provider | Notes |
|---|---|---|
| `gpt-5.2-2025-12-11` | OpenAI | Paper's primary judge |
| `gpt-5.4-2026-03-05` | OpenAI | Latest OpenAI judge |
| `gemini-3.1-pro-preview` | Google | Latest Gemini Pro judge |
| `gemini-3.1-flash-lite` | Google | Latest Gemini Flash judge |

Pass one or more judges via `--judge`. Results for each judge are cached and analyzed separately.

To register a new judge, add an entry to `JUDGE_REGISTRY` in the experiment file:

```python
"your-model-id": ("openai", "OPENAI_API_KEY"),  # or "google" + "GEMINI_API_KEY"
```

Then add the corresponding API key to `.env`.


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
@article{qi2026rift,
  title={RIFT: A RubrIc Failure Mode Taxonomy and Automated Diagnostics},
  author={Qi, Zhengyang and Dickens, Charles and Pham, Derek and Dsouza, Amanda and Parchami, Armin and Sala, Frederic and Varma, Paroma},
  journal={arXiv preprint arXiv:2604.01375},
  year={2026}
}
```
