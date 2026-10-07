"""
Generic RIFT runner — evaluate any JSONL rubric dataset with the shared pipeline.

Each line in the input file must have at minimum:
    input_context  : the prompt or conversation the rubric was written for
    rubric_text    : the rubric criteria to evaluate

Any additional fields are passed through as metadata in the output and can be
used as breakdown fields with --group-by.

Results are saved to results/run_<timestamp>.jsonl (cached per input file + judge).

Usage:
    uv run python run.py --input my_rubrics.jsonl
    uv run python run.py --input my_rubrics.jsonl --eval-strategy scoped --votes 3 --group-by domain
    uv run python run.py --input my_rubrics.jsonl --judge @openai/gpt-6-astra --concurrency 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from rift.experiment import Dataset, add_common_args, run_and_analyze
from rift.schema import Rubric

load_dotenv()


def load_rubrics(input_path: Path) -> list[Rubric]:
    rubrics = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "input_context" not in row or "rubric_text" not in row:
            sys.exit(f"Each record must have 'input_context' and 'rubric_text'. Offending row:\n{line}")
        metadata = {k: v for k, v in row.items() if k not in ("input_context", "rubric_text")}
        rubrics.append(Rubric(input_context=row["input_context"], rubric_text=row["rubric_text"], metadata=metadata))
    return rubrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_concurrency=10)
    parser.add_argument("--input", required=True, type=Path, help="JSONL file with input_context and rubric_text fields")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N rubrics")
    parser.add_argument("--group-by", nargs="*", default=[], metavar="FIELD", help="Metadata fields to break prevalence down by")
    args = parser.parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if args.eval_strategy is None:
        args.eval_strategy = "joined"  # flat JSONL has no criterion split; joined is the sensible default

    rubrics = load_rubrics(args.input)
    if args.n is not None:
        rubrics = rubrics[:args.n]
    print(f"Loaded {len(rubrics)} rubrics from {args.input.name}")
    ds = Dataset(
        name=args.input.stem,
        results_prefix="run",
        per_criterion=rubrics,
        per_rubric=rubrics,
        group_fields=args.group_by,
        cache_match={"dataset": args.input.stem},
        include_input_context=True,
    )
    asyncio.run(run_and_analyze(ds, args))
