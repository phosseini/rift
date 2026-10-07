"""
RIFT failure mode analysis on HealthBench Professional (525 conversations, 1,135 criteria).

Thin wrapper over the shared pipeline in rift/experiment.py (run from the repo root or anywhere; paths resolve to the repo): loads the dataset,
names the breakdown fields, and hands off. Strategy, caching, votes and the
console report are all shared with the other dataset scripts.

Results are cached to results/healthbench_pro_<timestamp>.jsonl. Scoped runs produce
eval_mode="per_criterion" and "per_conversation" records; joined runs "joined".

Usage:
    uv run python experiments/healthbench_pro.py --n 3 --concurrency 3 --judge gpt-5.4-2026-03-05   # smoke test
    uv run python experiments/healthbench_pro.py --concurrency 8 --eval-strategy scoped
    uv run python experiments/healthbench_pro.py --no-cache
"""

import argparse
import asyncio

from dotenv import load_dotenv

from rift.data.loaders import load_healthbench_professional
from rift.experiment import Dataset, add_common_args, run_and_analyze

load_dotenv()


def build_dataset(n: int | None) -> Dataset:
    print("Loading HealthBench Professional ...")
    per_criterion, per_conversation = load_healthbench_professional(n=n)
    return Dataset(
        name="healthbench_professional",
        results_prefix="healthbench_pro",
        per_criterion=per_criterion,
        per_rubric=per_conversation,
        rubric_eval_mode="per_conversation",
        group_fields=["use_case", "type", "difficulty", "sign"],
        unit="conversations",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_concurrency=2)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N conversations (default: all 525)")
    args = parser.parse_args()
    asyncio.run(run_and_analyze(build_dataset(args.n), args))
