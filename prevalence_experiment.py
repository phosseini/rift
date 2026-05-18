"""
Reproduces RIFT paper Table 2: failure mode prevalence in human-crafted vs. synthetic rubrics.

Human-crafted sources : advancedif, researchrubrics
Synthetic sources     : wildchecklists, openrubrics, autorubrics

Results are written incrementally to results/prevalence_<timestamp>.jsonl and cached
so the experiment can be re-run for analysis without repeating API calls.

Usage:
    uv run python prevalence_experiment.py --n 50 --concurrency 10
    uv run python prevalence_experiment.py --n 50 --no-cache
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from rift import ModelConfig, classify
from rift.data.loaders import (
    load_advancedif,
    load_autorubrics,
    load_openrubrics,
    load_researchrubrics,
    load_wildchecklists,
)
from rift.schema import Rubric
from rift.taxonomy import FAILURE_MODES

load_dotenv()

LABELS = [fm.label for fm in FAILURE_MODES]

SOURCES = [
    ("advancedif",      load_advancedif,      "human"),
    ("researchrubrics", load_researchrubrics,  "human"),
    ("wildchecklists",  load_wildchecklists,   "synthetic"),
    ("openrubrics",     load_openrubrics,      "synthetic"),
    ("autorubrics",     load_autorubrics,      "synthetic"),
]

# Reference values from RIFT paper Table 2
TABLE2 = {
    "human": {
        "subjective": 52.6, "non_atomic": 26.3, "ungrounded": 42.1,
        "misaligned_or_rigid": 63.2, "missing_criteria": 47.4,
        "hackable": 0.0, "low_signal": 21.1, "redundant_criteria": 26.3,
    },
    "synthetic": {
        "subjective": 86.7, "non_atomic": 60.0, "ungrounded": 46.7,
        "misaligned_or_rigid": 20.0, "missing_criteria": 36.7,
        "hackable": 13.3, "low_signal": 40.0, "redundant_criteria": 23.3,
    },
}

# ── result file helpers ───────────────────────────────────────────────────────

RESULTS_DIR = Path("results")


def _result_path(timestamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
    compact_ts = dt.strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"prevalence_{compact_ts}.jsonl"


def _load_cached(model: str) -> list[dict] | None:
    matches = sorted(RESULTS_DIR.glob("prevalence_*.jsonl"))
    for path in reversed(matches):
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        records = [json.loads(l) for l in lines]
        if records[0].get("judge_model") == model:
            sources = {r["source"] for r in records}
            expected = {s for s, _, _ in SOURCES}
            if expected.issubset(sources):
                print(f"  Cached: {path.name}  judge={model}  n={len(records)}")
                return records
    return None

# ── classification ────────────────────────────────────────────────────────────

async def run_classify(
    rubrics_by_source: dict[str, list[Rubric]],
    config: ModelConfig,
    concurrency: int,
    out_path: Path,
    n_votes: int = 1,
) -> list[dict]:
    fm_labels = LABELS
    semaphore = asyncio.Semaphore(concurrency)
    failed = 0

    flat: list[tuple[str, Rubric]] = [
        (source, rubric)
        for source, rubrics in rubrics_by_source.items()
        for rubric in rubrics
    ]

    async def classify_one(source: str, rubric: Rubric) -> dict:
        nonlocal failed
        async with semaphore:
            record = {
                "judge_model": config.model,
                "eval_mode": "joined",
                "included_failure_modes": fm_labels,
                "n_votes": n_votes,
                "source": source,
                **rubric.metadata,
                "rubric_text": rubric.rubric_text,
            }
            last_error = None
            for attempt in range(3):
                try:
                    result = await asyncio.wait_for(
                        classify(rubric, config, n_votes=n_votes), timeout=120
                    )
                    record.update({
                        "labels": sorted(lbl.label for lbl in result.labels),
                        "votes": [
                            [{"label": lbl.label, "justification": lbl.justification, "quote": lbl.quote} for lbl in run]
                            for run in result.votes
                        ],
                    })
                    last_error = None
                    break
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
            if last_error:
                failed += 1
                record.update({"labels": [], "votes": [], "error": last_error})
            return record

    tasks = [asyncio.create_task(classify_one(src, r)) for src, r in flat]
    records = []

    with open(out_path, "w") as f, tqdm(total=len(flat), unit="rubric", dynamic_ncols=True) as bar:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)
            bar.update(1)

    if failed:
        print(f"  [warn] {failed}/{len(flat)} calls failed — error field set in JSONL records.")

    return records

# ── analysis ──────────────────────────────────────────────────────────────────

def prevalence(records: list[dict], label: str) -> float:
    if not records:
        return 0.0
    return 100 * sum(1 for r in records if label in r["labels"]) / len(records)


def print_per_source_table(records: list[dict], n: int, model: str) -> None:
    short = {
        "advancedif": "ADV", "researchrubrics": "RES",
        "wildchecklists": "WC", "openrubrics": "OR", "autorubrics": "AR",
    }
    sources = [s for s, _, _ in SOURCES if any(r["source"] == s for r in records)]
    header = f"  {'':26}" + "".join(f"{short[s]:>7}" for s in sources)

    print(f"\n{'=' * 68}")
    print(f"  Failure mode prevalence by source  (n={n} per source, model: {model})")
    print(f"{'=' * 68}")
    print(header)
    print(f"  {'':-<26}" + "-" * (7 * len(sources)))
    for label in LABELS:
        row = f"  {label:<26}"
        for source in sources:
            subset = [r for r in records if r["source"] == source]
            row += f"{prevalence(subset, label):>6.0f}%"
        print(row)


def print_comparison_table(records: list[dict]) -> None:
    source_group = {s: g for s, _, g in SOURCES}
    human_records = [r for r in records if source_group.get(r["source"]) == "human"]
    synth_records  = [r for r in records if source_group.get(r["source"]) == "synthetic"]

    print(f"\n{'=' * 68}")
    print(f"  Human vs Synthetic  (comparison to RIFT paper Table 2)")
    print(f"{'=' * 68}")
    print(f"  {'':26}{'Ours/H':>8}{'Paper/H':>8}  {'Ours/S':>8}{'Paper/S':>8}")
    print(f"  {'':-<26}" + "-" * 34)
    for label in LABELS:
        our_h  = prevalence(human_records, label)
        our_s  = prevalence(synth_records, label)
        pap_h  = TABLE2["human"][label]
        pap_s  = TABLE2["synthetic"][label]
        print(f"  {label:<26}{our_h:>7.1f}%{pap_h:>7.1f}%  {our_s:>7.1f}%{pap_s:>7.1f}%")

# ── main ─────────────────────────────────────────────────────────────────────

JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    "gpt-5.2-2025-12-11":         ("openai", "OPENAI_API_KEY"),
    "gpt-5.4-2026-03-05":         ("openai", "OPENAI_API_KEY"),
    "gemini-3.1-pro-preview":     ("google", "GEMINI_API_KEY"),
    "gemini-3.1-flash-lite":      ("google", "GEMINI_API_KEY"),
}

DEFAULT_JUDGES = ["gpt-5.4-2026-03-05", "gemini-3.1-pro-preview"]


def build_configs(judges: list[str]) -> list[ModelConfig]:
    configs = []
    for model in judges:
        if model not in JUDGE_REGISTRY:
            sys.exit(f"Unknown judge '{model}'. Known: {list(JUDGE_REGISTRY)}")
        provider, env_var = JUDGE_REGISTRY[model]
        key = os.getenv(env_var) or sys.exit(f"{env_var} not set in .env")
        configs.append(ModelConfig(model, provider, key))
    return configs


async def main(n: int, concurrency: int, no_cache: bool, judges: list[str], n_votes: int = 1) -> None:
    configs = build_configs(judges)

    print(f"Loading {n} rubrics from each of {len(SOURCES)} sources ...")
    rubrics_by_source: dict[str, list[Rubric]] = {}
    for source, loader, _ in SOURCES:
        rubrics = loader(n)
        rubrics_by_source[source] = rubrics
        print(f"  {source:<22} {len(rubrics)} rubrics loaded")

    for config in configs:
        cached = None if no_cache else _load_cached(config.model)

        if cached is not None:
            print(f"\nLoaded cached results for {config.model}  ({len(cached)} records)")
            print_per_source_table(cached, n, config.model)
            print_comparison_table(cached)
        else:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
            path = _result_path(timestamp)
            print(f"\nClassifying with {config.model} ...")
            records = await run_classify(rubrics_by_source, config, concurrency, path, n_votes=n_votes)
            print(f"  Results saved to {path}")
            print_per_source_table(records, n, config.model)
            print_comparison_table(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Number of rubrics per source (default: 10)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max simultaneous API calls (default: 10)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-run even if cached results exist")
    parser.add_argument("--judge", nargs="+", default=DEFAULT_JUDGES, metavar="MODEL",
                        help=f"Judge model(s). Known: {list(JUDGE_REGISTRY)}. Default: {DEFAULT_JUDGES}")
    parser.add_argument("--votes", type=int, default=1,
                        help="Number of judge runs per rubric; majority vote used when >1 (default: 1)")
    args = parser.parse_args()
    asyncio.run(main(args.n, args.concurrency, args.no_cache, args.judge, args.votes))
