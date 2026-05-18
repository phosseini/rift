"""
RIFT failure mode analysis on HealthBench Professional.

Evaluation strategy is set by eval_strategy in config.json:

  "joined" — all criteria for each conversation are concatenated into one string
             and evaluated with all enabled failure modes together.
             Equivalent to how AdvancedIF, WildChecklists etc. are run in the paper.
             (eval_mode="joined")

  "scoped" — each failure mode is applied at its natural scope:
             criterion-scope modes on each criterion (eval_mode="per_criterion")
             rubric-scope modes on the joined full rubric (eval_mode="per_conversation")

Which failure modes are active is controlled by config.json
("enabled_failure_modes" key). Remove a label to exclude it entirely.

Results are cached to results/hbp_<timestamp>.jsonl. Each row records
eval_mode and included_failure_modes so the file is self-describing.

Usage:
    uv run python hbp_experiment.py                  # default judges from config
    uv run python hbp_experiment.py --concurrency 5  # control concurrency
    uv run python hbp_experiment.py --no-cache       # force re-run even if cached
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from rift import ModelConfig, classify
from rift.data.loaders import load_healthbench_professional
from rift.schema import Rubric
from rift.taxonomy import FAILURE_MODES, FailureMode

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config.json")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {
        "eval_strategy": "joined",
        "failure_modes": {fm.label: fm.scope for fm in FAILURE_MODES},
    }


def resolve_failure_modes(
    failure_modes_cfg: dict[str, str],
) -> tuple[list[FailureMode], list[FailureMode]]:
    """Return (criterion_fms, rubric_fms) from the merged config dict.

    Keys are enabled failure mode labels; values are their scopes.
    Scope in config takes precedence over the default in taxonomy.py.
    """
    criterion_fms = [
        fm for fm in FAILURE_MODES
        if fm.label in failure_modes_cfg and failure_modes_cfg[fm.label] == "criterion"
    ]
    rubric_fms = [
        fm for fm in FAILURE_MODES
        if fm.label in failure_modes_cfg and failure_modes_cfg[fm.label] == "rubric"
    ]
    return criterion_fms, rubric_fms

# ── result file helpers ───────────────────────────────────────────────────────

RESULTS_DIR = Path("results")


def _result_path(timestamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
    compact_ts = dt.strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"hbp_{compact_ts}.jsonl"


def _expected_modes(strategy: str) -> set[str]:
    if strategy == "joined":
        return {"joined"}
    return {"per_criterion", "per_conversation"}  # "scoped"


def _load_cached(model: str, strategy: str) -> list[dict] | None:
    """Load the most recent result file for this model that covers the required eval modes."""
    required = _expected_modes(strategy)
    matches = sorted(RESULTS_DIR.glob("hbp_*.jsonl"))
    for path in reversed(matches):
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        records = [json.loads(l) for l in lines]
        if records[0].get("judge_model") != model:
            continue
        if required.issubset({r.get("eval_mode") for r in records}):
            r = records[0]
            print(f"  Cached: {path.name}  dataset={r['dataset']}  judge_model={r['judge_model']}  n={len(records)}")
            return records
    return None

# ── classification ────────────────────────────────────────────────────────────

async def run_classify(
    rubrics: list[Rubric],
    config: ModelConfig,
    failure_modes: list[FailureMode],
    concurrency: int,
    out_path: Path,
    dataset: str,
    append: bool = False,
    n_votes: int = 1,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    fm_labels = sorted(fm.label for fm in failure_modes)

    async def classify_one(rubric: Rubric) -> tuple[Rubric, set[str], list[list[str]], str | None]:
        async with semaphore:
            last_error = None
            for attempt in range(3):
                try:
                    result = await asyncio.wait_for(
                        classify(rubric, config, failure_modes=failure_modes, n_votes=n_votes),
                        timeout=120,
                    )
                    votes = [
                        [{"label": lbl.label, "justification": lbl.justification, "quote": lbl.quote} for lbl in run]
                        for run in result.votes
                    ]
                    return rubric, {lbl.label for lbl in result.labels}, votes, None
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
            return rubric, set(), [], last_error

    total = len(rubrics)
    eval_mode = rubrics[0].metadata.get("eval_mode", "unknown") if rubrics else "unknown"
    print(f"  [{eval_mode}] Classifying {total} rubrics with {config.model} (concurrency={concurrency}) ...")

    tasks = [asyncio.create_task(classify_one(r)) for r in rubrics]
    records = []
    failed = 0

    write_mode = "a" if append else "w"
    with open(out_path, write_mode) as f, tqdm(total=total, unit="rubric", dynamic_ncols=True) as bar:
        for coro in asyncio.as_completed(tasks):
            rubric, labels, votes, error = await coro
            record = {
                "dataset": dataset,
                "judge_model": config.model,
                **rubric.metadata,
                "rubric_text": rubric.rubric_text,
                "labels": sorted(labels),
                "votes": votes,
                "included_failure_modes": fm_labels,
                "n_votes": n_votes,
            }
            if error:
                record["error"] = error
                failed += 1
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)
            bar.update(1)

    if failed:
        print(f"  [warn] {failed}/{total} calls failed — error field set in JSONL records.")

    return records

# ── analysis helpers ──────────────────────────────────────────────────────────

def prevalence(records: list[dict], label: str) -> float:
    if not records:
        return 0.0
    return 100 * sum(1 for r in records if label in r["labels"]) / len(records)


def print_table(
    title: str,
    groups: dict[str, list[dict]],
    labels: list[str],
    min_width: int = 22,
) -> None:
    col_w = max(min_width, max(len(k) for k in groups))
    header = f"  {'':26}" + "".join(f"{k:>{col_w}}" for k in groups)
    sep = "  " + "-" * (26 + col_w * len(groups))
    width = max(len(header), 68)

    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    print(header)
    print(sep)
    for label in labels:
        row = f"  {label:<26}"
        for records in groups.values():
            pct = prevalence(records, label)
            row += f"{pct:>{col_w-1}.0f}%" if pct > 0 else f"{'—':>{col_w}}"
        print(row)
    print(f"  {'':26}" + "".join(f"{len(v):>{col_w}}" for v in groups.values()))
    print(f"  {'':26}" + "".join(f"{'(n)':>{col_w}}" for _ in groups))


def print_flagged_examples(records: list[dict], labels: list[str], n_per_mode: int = 2) -> None:
    print(f"\n{'=' * 68}")
    eval_mode = records[0].get("eval_mode", "unknown") if records else "unknown"
    print(f"  Flagged examples [{eval_mode}] (up to {n_per_mode} per failure mode)")
    print(f"{'=' * 68}")
    for label in labels:
        flagged = [r for r in records if label in r["labels"]]
        if not flagged:
            continue
        print(f"\n  [{label}]  — {len(flagged)} rubrics flagged")
        for r in flagged[:n_per_mode]:
            if eval_mode == "per_criterion":
                pts = r["points"]
                print(f"    [{pts:+d} pts | {r['use_case']} | {r['specialty']}]")
                print(f"    {r['criterion_text'][:140]}")
            else:  # "joined" or "per_conversation"
                print(f"    [{r['criterion_count']} criteria | {r['use_case']} | {r['specialty']}]")


def analyze_level(records: list[dict], model: str, level_label: str) -> None:
    if not records:
        return
    labels = records[0].get("included_failure_modes", [])

    print_table(
        f"Overall prevalence [{level_label}]  (n={len(records)}, model: {model})",
        {level_label: records},
        labels,
        min_width=16,
    )
    print_table(
        f"By use_case [{level_label}]",
        {uc: [r for r in records if r["use_case"] == uc]
         for uc in ["consult", "research", "writing"]},
        labels,
    )
    print_table(
        f"By type [{level_label}]",
        {t: [r for r in records if r["type"] == t]
         for t in ["good_faith", "red_teaming"]},
        labels,
    )
    print_table(
        f"By difficulty [{level_label}]",
        {d: [r for r in records if r["difficulty"] == d]
         for d in ["typical", "difficult"]},
        labels,
    )
    if level_label == "per_criterion":
        print_table(
            "By criterion sign",
            {
                "positive (+pts)": [r for r in records if r.get("points", 0) > 0],
                "negative (−pts)": [r for r in records if r.get("points", 0) < 0],
            },
            labels,
        )
    print_flagged_examples(records, labels)


def analyze(records: list[dict], model: str) -> None:
    seen_modes = {r.get("eval_mode") for r in records}
    for mode in ("joined", "per_criterion", "per_conversation"):
        if mode in seen_modes:
            analyze_level([r for r in records if r.get("eval_mode") == mode], model, mode)

# ── main ─────────────────────────────────────────────────────────────────────

JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    "gpt-5.2-2025-12-11":         ("openai", "OPENAI_API_KEY"),
    "gpt-5.4-2026-03-05":         ("openai", "OPENAI_API_KEY"),
    "gemini-3.1-pro-preview":     ("google", "GEMINI_API_KEY"),
    "gemini-3.1-flash-lite":      ("google", "GEMINI_API_KEY"),
}

DEFAULT_JUDGES = ["gpt-5.4-2026-03-05"]


def build_configs(judges: list[str]) -> list[ModelConfig]:
    configs = []
    for model in judges:
        if model not in JUDGE_REGISTRY:
            sys.exit(f"Unknown judge '{model}'. Known judges: {list(JUDGE_REGISTRY)}")
        provider, env_var = JUDGE_REGISTRY[model]
        key = os.getenv(env_var) or sys.exit(f"{env_var} not set in .env")
        configs.append(ModelConfig(model, provider, key))
    return configs


async def main(concurrency: int, no_cache: bool, judges: list[str], n: int | None = None, eval_strategy: str | None = None, n_votes: int = 1) -> None:
    cfg = load_config()
    strategy = eval_strategy or cfg.get("eval_strategy", "scoped")
    fm_cfg = cfg.get("failure_modes", {fm.label: fm.scope for fm in FAILURE_MODES})
    all_fms = [fm for fm in FAILURE_MODES if fm.label in fm_cfg]
    criterion_fms, rubric_fms = resolve_failure_modes(fm_cfg)

    print(f"Strategy: {strategy}")
    print(f"Enabled failure modes: {[fm.label for fm in all_fms]}")
    if strategy == "scoped":
        print(f"  criterion-scope: {[fm.label for fm in criterion_fms]}")
        print(f"  rubric-scope:    {[fm.label for fm in rubric_fms]}")

    configs = build_configs(judges)

    print("Loading HealthBench Professional ...")
    per_criterion, per_conversation = load_healthbench_professional(n=n)
    print(f"  {len(per_criterion)} criteria across {len(per_conversation)} conversations")

    for config in configs:
        cached = None if no_cache else _load_cached(config.model, strategy)

        if cached is not None:
            print(f"\nLoaded cached results for {config.model}  ({len(cached)} records)")
            analyze(cached, config.model)
            continue

        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        path = _result_path(timestamp)
        records = []
        appending = False

        # ── joined pass (paper-equivalent: all criteria joined, all modes) ───
        if strategy == "joined" and all_fms:
            import copy
            joined_rubrics = [copy.copy(r) for r in per_conversation]
            for jr in joined_rubrics:
                jr.metadata = {**jr.metadata, "eval_mode": "joined"}
            records += await run_classify(
                joined_rubrics, config, all_fms, concurrency, path,
                "healthbench_professional", append=appending, n_votes=n_votes,
            )
            appending = True

        # ── scoped pass (failure modes applied at their natural scope) ───────
        if strategy == "scoped":
            if criterion_fms:
                records += await run_classify(
                    per_criterion, config, criterion_fms, concurrency, path,
                    "healthbench_professional", append=appending, n_votes=n_votes,
                )
                appending = True
            if rubric_fms:
                records += await run_classify(
                    per_conversation, config, rubric_fms, concurrency, path,
                    "healthbench_professional", append=appending, n_votes=n_votes,
                )

        print(f"  Results saved to {path}")
        analyze(records, config.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--judge", nargs="+", default=DEFAULT_JUDGES, metavar="MODEL")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N conversations (default: all 525)")
    parser.add_argument("--eval-strategy", choices=["joined", "scoped"], default=None,
                        help="Override eval_strategy from config.json")
    parser.add_argument("--votes", type=int, default=1,
                        help="Number of judge runs per rubric; majority vote used when >1 (default: 1)")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.no_cache, args.judge, args.n, args.eval_strategy, args.votes))
