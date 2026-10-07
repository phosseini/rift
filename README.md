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

`.env` keys — provide **either** direct provider keys **or** a Portkey key (or both):
```
OPENAI_API_KEY=...        # direct OpenAI
GEMINI_API_KEY=...        # direct Google
PORTKEY_API_KEY=...       # Portkey gateway, used when a direct key is missing or when --judge names a @catalog/address
PORTKEY_BASE_URL=https://api.portkey.ai/v1
```
See [Judges](#judges) for how judges are resolved to credentials. Everything else the pipeline reads (default judge, evaluation strategy, failure-mode scopes, model prices) lives in [`config/rift.yaml`](#configuration-configriftyaml).

Check that credentials and the judge route work before spending on a run:

```bash
uv run python sanity_check.py --judge gpt-5.4-2026-03-05   # 16 taxonomy examples, reports hits per failure mode
```


## Quick smoke test

```bash
# Prevalence experiment — 3 rubrics per source, fast sanity check
uv run python experiments/prevalence.py --n 3 --concurrency 5 --judge gpt-5.4-2026-03-05

# HealthBench Professional experiment — 3 conversations
uv run python experiments/healthbench_pro.py --n 3 --concurrency 3 --judge gpt-5.4-2026-03-05 --eval-strategy scoped

```


## Repository layout

```
rift/                     the library
  taxonomy.py             the eight failure modes: descriptions, scopes, pass/fail examples
  prompts.py              judge prompt built from the taxonomy
  classifier.py           one judge call per rubric (direct OpenAI/Google via hopper, or Portkey), majority voting, token usage
  judges.py               resolves --judge values to credentials and routes
  config.py               loads config/rift.yaml plus an optional overlay
  experiment.py           shared runner: Dataset, caching, run manifests and cost, console analysis
  data/loaders.py         loaders for the paper datasets and HealthBench Professional
config/rift.yaml          evaluation settings, judge registry, Portkey settings, pricing
experiments/
  prevalence.py           paper Table 2 over five rubric datasets
  healthbench_pro.py      HealthBench Professional
run.py                    any JSONL rubric file (see Bring your own dataset)
sanity_check.py           taxonomy pass/fail examples against a judge
notebooks/                analysis notebooks over results/*.jsonl
results/                  run outputs (<prefix>_<timestamp>.jsonl + .meta.json), gitignored
```


## Experiments

### 1. Prevalence experiment (`experiments/prevalence.py`)

Reproduces RIFT paper Table 2. Evaluates failure mode prevalence across five rubric datasets using the **joined** strategy (full rubric evaluated with all failure modes — paper-equivalent method).

```bash
uv run python experiments/prevalence.py --n 50 --concurrency 10 --judge gpt-5.2-2025-12-11
```

Results are saved to `results/prevalence_<timestamp>.jsonl` and cached per judge. Each record includes `rubric_text`, `labels` (majority-voted), `votes` (raw per-run outputs), `n_votes`, and an `error` field if the API call failed.

Default for `--n` is `10` (5 sources → 50 total API calls). The paper uses `--votes 5`.


### 2. HealthBench Professional experiment (`experiments/healthbench_pro.py`)

Runs RIFT on all 525 conversations and 1,135 rubric criteria from [HealthBench Professional](https://huggingface.co/datasets/openai/healthbench-professional).

```bash
uv run python experiments/healthbench_pro.py --concurrency 8 --judge gpt-5.4-2026-03-05 --eval-strategy scoped
```

Results are saved to `results/healthbench_pro_<timestamp>.jsonl` and cached per judge + strategy. Each record includes `rubric_text`, `labels` (majority-voted), `votes` (raw per-run outputs with the judge's justification and quote), `n_votes`, `usage` (token counts), and an `error` field if the call failed or the judge's reply was truncated or unparseable after retries; a cut-off reply is never recorded as "no failure modes".

`--eval-strategy` controls how failure modes are applied:

- **`joined`** — all rubric criteria for a conversation are concatenated into one string and evaluated with all failure modes together. Equivalent to the paper's method.
- **`scoped`** (default) — criterion-scope failure modes run on each criterion individually; rubric-scope modes run on the full joined rubric. Produces `per_criterion` and `per_conversation` records, letting you pinpoint failure modes at the criterion level rather than just the rubric.


## Run manifests and cost

Every run writes a sidecar next to its results file, `results/<run_id>.meta.json`, first when the run starts and again when it finishes. It records everything needed to reproduce or account for the run:

- judge name, provider, Portkey wire model and base URL, JSON mode
- strategy, votes, concurrency, the failure-mode config, the config overlay in use (if any), the full CLI arguments and command line
- dataset name and counts, the input-context mode
- git commit and whether the tree was dirty (so prompt edits can be tied to runs)
- start and end time, duration, calls, errors, records per eval mode
- token totals (input, output, cached input) and the USD cost

Cost is computed from the `pricing` table in `config/rift.yaml` (USD per 1M tokens, keyed by model). Fill in rates from the provider's price list or the negotiated rate your Portkey workspace shows; a model with `null` rates still gets its tokens recorded and the manifest marks cost as incomplete. Every Portkey request is tagged with `x-portkey-metadata` carrying the `run_id`, so the same run can be filtered in the Portkey logs to cross-check the figure. Each result record also carries its own `usage`, so cost can be broken down by task type, role, or annotator in the notebooks.

The cache check also compares record counts against the dataset, so a `--n` smoke test is never mistaken for a full run.


## Adding a dataset

All dataset scripts are thin wrappers over one shared pipeline, `rift/experiment.py`, which owns config handling, the concurrent classification loop with retries, incremental JSONL writing, result caching, and the console prevalence tables. A new dataset needs two things:

1. A loader in `rift/data/loaders.py` returning `(per_criterion, per_rubric)` lists of `Rubric` objects, with whatever metadata you want to break results down by.
2. A short script that builds a `Dataset(name, results_prefix, per_criterion, per_rubric, group_fields=[...])` and calls `run_and_analyze`. See `experiments/healthbench_pro.py` as the template; put the script in `experiments/`.

Do not add dataset-specific logic to the shared module; use `group_fields`, `cache_match`, and metadata instead. Result records share one schema across datasets, so the notebooks' loading code is a reusable starting point.


## Bring your own dataset

To run RIFT on any rubric dataset, prepare a JSONL file where each line has two required fields:

```jsonl
{"input_context": "Write a haiku about winter.", "rubric_text": "5 pts: Contains exactly 17 syllables in 5-7-5 structure."}
{"input_context": "Summarize the article.", "rubric_text": "10 pts: Covers all main points accurately."}
```

Any additional fields are passed through as metadata in the output. Then run:

```bash
uv run python run.py --input my_rubrics.jsonl
uv run python run.py --input my_rubrics.jsonl --eval-strategy scoped --votes 3 --judge gpt-5.4-2026-03-05
uv run python run.py --input my_rubrics.jsonl --group-by domain   # break prevalence down by a metadata field
```

A sample file with 10 rubrics drawn from the five paper datasets is included for quick testing:

```bash
uv run python run.py --input sample_rubrics.jsonl --concurrency 5 --judge gpt-5.4-2026-03-05
```

Results are saved to `results/run_<timestamp>.jsonl` with the same schema as the other experiments.


## Parameters

All experiments share the same CLI parameters:

```
--judge            judges.default in config   Judge model(s), space-separated: registered id, @portkey/catalog/address, or portkey:<model>.
--concurrency      (experiment-specific)      Max simultaneous API calls. Lower this if you hit rate limits.
--n                (experiment-specific)      Limit to first N items (rubrics or conversations). Useful for quick tests.
--votes            evaluation.votes (1)       Judge runs per rubric; majority vote when >1. The paper uses 5.
--eval-strategy    evaluation.strategy        joined or scoped — see the HealthBench Professional experiment section. run.py defaults to joined.
--no-cache         off                        Force re-run even if cached results exist for this judge + strategy.
--config           none                       Overlay YAML merged over config/rift.yaml (or set RIFT_CONFIG).
```

Defaults in the middle column come from `config/rift.yaml` unless noted. `experiments/prevalence.py` keeps its own defaults (two judges, joined strategy, 10 rubrics per source) to match the paper's setup.


## Datasets

| Dataset | HuggingFace | Type | Used in |
|---|---|---|---|
| AdvancedIF | 🤗 [facebook/AdvancedIF](https://huggingface.co/datasets/facebook/AdvancedIF) | Human-curated | Prevalence experiment |
| ResearchRubrics | 🤗 [ScaleAI/researchrubrics](https://huggingface.co/datasets/ScaleAI/researchrubrics) | Human-written | Prevalence experiment |
| WildChecklists | 🤗 [viswavi/wildchecklists](https://huggingface.co/datasets/viswavi/wildchecklists) | LLM-generated | Prevalence experiment |
| OpenRubrics | 🤗 [OpenRubrics/OpenRubrics](https://huggingface.co/datasets/OpenRubrics/OpenRubrics) | LLM-generated | Prevalence experiment |
| Auto-Rubric | 🤗 [agentscope-ai/Auto-Rubric](https://huggingface.co/datasets/agentscope-ai/Auto-Rubric) | LLM-generated | Prevalence experiment |
| HealthBench Professional | 🤗 [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) | Physician-written | HealthBench Professional experiment |


## Judges

Registered ids (`judges.registry` in `config/rift.yaml`):

| Model ID | Provider | Notes |
|---|---|---|
| `gpt-5.2-2025-12-11` | OpenAI | Paper's primary judge |
| `gpt-5.4-2026-03-05` | OpenAI | Latest OpenAI judge |
| `gemini-3.1-pro-preview` | Google | Latest Gemini Pro judge |
| `gemini-3.1-flash-lite` | Google | Latest Gemini Flash judge |

Pass one or more judges via `--judge`. Results for each judge are cached and analyzed separately. A registered id uses the provider's direct key from `.env`, or is routed through Portkey when only `PORTKEY_API_KEY` is set; `--judge` also accepts any Portkey catalog address directly (`@slug/model`).

To register a new **direct** judge, add an entry under `judges.registry` in `config/rift.yaml` (or in an overlay file):

```yaml
judges:
  registry:
    your-model-id: {provider: openai, key_env: OPENAI_API_KEY}   # or provider: google, key_env: GEMINI_API_KEY
```


## Analysis notebooks

`notebooks/` contains one notebook per experiment; each auto-loads the matching `results/*.jsonl` files and indexes them by judge (and strategy):

| Notebook | Results glob | Extra views |
|---|---|---|
| `prevalence_analysis.ipynb` | `prevalence_*.jsonl` | per-source prevalence vs paper Table 2, co-occurrence |
| `healthbench_professional_analysis.ipynb` | `healthbench_pro_*.jsonl` | by use case / type / difficulty, joined vs scoped, cross-judge kappa |

```bash
uv sync --group notebooks && uv run jupyter lab notebooks/
```


## Configuration (`config/rift.yaml`)

One YAML file holds everything non-secret the pipeline reads; API keys stay in `.env`. It is resolved relative to the repo root, so scripts work from any directory.

```yaml
evaluation:
  strategy: scoped          # joined | scoped   (--eval-strategy overrides)
  votes: 1                  # judge runs per record, majority label when > 1   (--votes overrides)
  failure_modes:            # scope each mode is judged at; remove a mode to disable it everywhere
    subjective: criterion
    missing_criteria: rubric
    # ...

judges:
  default: gpt-5.4-2026-03-05
  registry:                 # ids accepted by --judge; provider picks the client, key_env the .env variable
    gpt-5.4-2026-03-05: {provider: openai, key_env: OPENAI_API_KEY}
  portkey:
    base_url: https://api.portkey.ai/v1
    slugs: {openai: "@openai", google: "@google"}   # catalog slugs used when falling back to Portkey

pricing:                    # USD per 1M tokens; null = unknown, cost marked incomplete in the manifest
  gpt-6-astra: {input_per_1m: 10.0, output_per_1m: 50.0, cached_input_per_1m: null}
```

**Overlays.** Pass `--config my.yaml` to any experiment script, or set `RIFT_CONFIG=my.yaml`, to deep-merge a second file over the default. Use it for workspace-specific judges, catalog slugs or negotiated prices without editing the shared file; the manifest records which overlay was active.

## Reference

```bibtex
@article{qi2026rift,
  title={RIFT: A RubrIc Failure Mode Taxonomy and Automated Diagnostics},
  author={Qi, Zhengyang and Dickens, Charles and Pham, Derek and Dsouza, Amanda and Parchami, Armin and Sala, Frederic and Varma, Paroma},
  journal={arXiv preprint arXiv:2604.01375},
  year={2026}
}
```
