"""
Generic RIFT runner — evaluate any JSONL rubric dataset.

Each line in the input file must have at minimum:
    input_context  : the prompt or conversation the rubric was written for
    rubric_text    : the rubric criteria to evaluate

Any additional fields are passed through as metadata in the output.

Results are saved to results/run_<timestamp>.jsonl.

Usage:
    uv run python run.py --input my_rubrics.jsonl
    uv run python run.py --input my_rubrics.jsonl --eval-strategy scoped --votes 3
    uv run python run.py --input my_rubrics.jsonl --judge gemini-3.1-flash-lite --concurrency 10
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
from rift.schema import Rubric
from rift.taxonomy import FAILURE_MODES, FailureMode

load_dotenv()

CONFIG_PATH = Path("config.json")
RESULTS_DIR = Path("results")

JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    "gpt-5.2-2025-12-11":     ("openai", "OPENAI_API_KEY"),
    "gpt-5.4-2026-03-05":     ("openai", "OPENAI_API_KEY"),
    "gemini-3.1-pro-preview": ("google", "GEMINI_API_KEY"),
    "gemini-3.1-flash-lite":  ("google", "GEMINI_API_KEY"),
}

DEFAULT_JUDGE = "gpt-5.4-2026-03-05"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"failure_modes": {fm.label: fm.scope for fm in FAILURE_MODES}}


def resolve_failure_modes(fm_cfg: dict[str, str]) -> tuple[list[FailureMode], list[FailureMode]]:
    criterion_fms = [fm for fm in FAILURE_MODES if fm_cfg.get(fm.label) == "criterion"]
    rubric_fms    = [fm for fm in FAILURE_MODES if fm_cfg.get(fm.label) == "rubric"]
    return criterion_fms, rubric_fms


def load_rubrics(input_path: Path) -> list[Rubric]:
    rubrics = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "input_context" not in row or "rubric_text" not in row:
            sys.exit(f"Each record must have 'input_context' and 'rubric_text'. Offending row:\n{line}")
        metadata = {k: v for k, v in row.items() if k not in ("input_context", "rubric_text")}
        rubrics.append(Rubric(
            input_context=row["input_context"],
            rubric_text=row["rubric_text"],
            metadata=metadata,
        ))
    return rubrics


async def run_classify(
    rubrics: list[Rubric],
    config: ModelConfig,
    failure_modes: list[FailureMode],
    concurrency: int,
    out_path: Path,
    eval_mode: str,
    append: bool = False,
    n_votes: int = 1,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    fm_labels = sorted(fm.label for fm in failure_modes)

    async def classify_one(rubric: Rubric) -> tuple[Rubric, set[str], list, str | None]:
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

    print(f"  [{eval_mode}] Classifying {len(rubrics)} rubrics with {config.model} (concurrency={concurrency}) ...")
    tasks = [asyncio.create_task(classify_one(r)) for r in rubrics]
    records = []
    failed = 0

    write_mode = "a" if append else "w"
    with open(out_path, write_mode) as f, tqdm(total=len(rubrics), unit="rubric", dynamic_ncols=True) as bar:
        for coro in asyncio.as_completed(tasks):
            rubric, labels, votes, error = await coro
            record = {
                "judge_model": config.model,
                "eval_mode": eval_mode,
                **rubric.metadata,
                "input_context": rubric.input_context,
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
        print(f"  [warn] {failed}/{len(rubrics)} calls failed — error field set in output.")
    return records


def print_results(records: list[dict], eval_mode: str) -> None:
    if not records:
        return
    fm_labels = records[0].get("included_failure_modes", [])
    print(f"\n{'=' * 60}")
    print(f"  Failure mode prevalence [{eval_mode}]  (n={len(records)})")
    print(f"{'=' * 60}")
    for fm in fm_labels:
        pct = 100 * sum(1 for r in records if fm in r["labels"]) / len(records)
        bar = "█" * int(pct / 2)
        print(f"  {fm:<26} {pct:5.1f}%  {bar}")


async def main(
    input_path: Path,
    judges: list[str],
    concurrency: int,
    eval_strategy: str,
    n_votes: int,
    n: int | None,
) -> None:
    cfg = load_config()
    fm_cfg = cfg.get("failure_modes", {fm.label: fm.scope for fm in FAILURE_MODES})
    criterion_fms, rubric_fms = resolve_failure_modes(fm_cfg)
    all_fms = criterion_fms + rubric_fms

    print(f"Input:    {input_path}")
    print(f"Strategy: {eval_strategy}")
    print(f"Enabled failure modes: {[fm.label for fm in all_fms]}")

    rubrics = load_rubrics(input_path)
    if n is not None:
        rubrics = rubrics[:n]
    print(f"Loaded {len(rubrics)} rubrics from {input_path.name}")

    for model in judges:
        if model not in JUDGE_REGISTRY:
            sys.exit(f"Unknown judge '{model}'. Known: {list(JUDGE_REGISTRY)}")
        provider, env_var = JUDGE_REGISTRY[model]
        key = os.getenv(env_var) or sys.exit(f"{env_var} not set in .env")
        config = ModelConfig(model, provider, key)

        RESULTS_DIR.mkdir(exist_ok=True)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"run_{ts}.jsonl"

        if eval_strategy == "joined":
            records = await run_classify(
                rubrics, config, all_fms, concurrency, out_path, "joined", n_votes=n_votes,
            )
            print_results(records, "joined")

        elif eval_strategy == "scoped":
            records = []
            appending = False
            if criterion_fms:
                r = await run_classify(
                    rubrics, config, criterion_fms, concurrency, out_path,
                    "per_criterion", append=appending, n_votes=n_votes,
                )
                records += r
                appending = True
            if rubric_fms:
                r = await run_classify(
                    rubrics, config, rubric_fms, concurrency, out_path,
                    "per_rubric", append=appending, n_votes=n_votes,
                )
                records += r
            print_results([r for r in records if r["eval_mode"] == "per_criterion"], "per_criterion")
            print_results([r for r in records if r["eval_mode"] == "per_rubric"], "per_rubric")

        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RIFT on any JSONL rubric dataset.")
    parser.add_argument("--input", required=True, type=Path,
                        help="JSONL file with input_context and rubric_text fields")
    parser.add_argument("--judge", nargs="+", default=[DEFAULT_JUDGE], metavar="MODEL",
                        help=f"Judge model(s), space-separated (default: {DEFAULT_JUDGE})")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max simultaneous API calls (default: 10)")
    parser.add_argument("--eval-strategy", choices=["joined", "scoped"], default="joined",
                        help="joined: all modes on full rubric. scoped: criterion/rubric modes separately (default: joined)")
    parser.add_argument("--votes", type=int, default=1,
                        help="Number of judge runs per rubric; majority vote when >1 (default: 1)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N rubrics (default: all)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    asyncio.run(main(args.input, args.judge, args.concurrency, args.eval_strategy, args.votes, args.n))
