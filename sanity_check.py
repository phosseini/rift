"""
Sanity check for the RIFT implementation using the illustrative examples from Appendix D.

For each of the 8 failure modes the taxonomy provides:
  - A PASS example: the rubric does NOT exhibit this failure mode → should NOT be detected
  - A FAIL example: the rubric DOES exhibit this failure mode   → SHOULD be detected

We run classify() on all 16 examples and report per-case results and an overall score.
Run with one or both API keys set; whichever models are configured will be tested.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from rift import ModelConfig, Rubric, classify
from rift.taxonomy import FAILURE_MODES

load_dotenv()


@dataclass
class TestCase:
    failure_mode: str
    rubric: Rubric
    should_detect: bool  # True for FAIL examples, False for PASS examples


def build_test_cases() -> list[TestCase]:
    cases = []
    for fm in FAILURE_MODES:
        for ex in fm.pass_examples:
            cases.append(TestCase(
                failure_mode=fm.label,
                rubric=Rubric(rubric_text=ex["rubric"], input_context=ex["input_context"]),
                should_detect=False,
            ))
        for ex in fm.fail_examples:
            cases.append(TestCase(
                failure_mode=fm.label,
                rubric=Rubric(rubric_text=ex["rubric"], input_context=ex["input_context"]),
                should_detect=True,
            ))
    return cases


async def run_cases(cases: list[TestCase], config: ModelConfig) -> list[set[str]]:
    semaphore = asyncio.Semaphore(10)

    async def classify_one(case):
        async with semaphore:
            return await classify(case.rubric, config)

    results = await asyncio.gather(*[classify_one(c) for c in cases])
    return [{label.label for label in result.labels} for result in results]


def print_results(cases: list[TestCase], detected: list[set[str]], model: str) -> int:
    print(f"\n{'=' * 62}")
    print(f"  Model: {model}")
    print(f"{'=' * 62}")

    fail_cases = [(c, d) for c, d in zip(cases, detected) if c.should_detect]
    pass_cases = [(c, d) for c, d in zip(cases, detected) if not c.should_detect]

    n_correct = 0

    print("\nFAIL examples — classifier SHOULD detect the failure mode:")
    for case, found in fail_cases:
        ok = case.failure_mode in found
        n_correct += ok
        mark = "✓" if ok else "✗"
        extra = f"  (got: {sorted(found)})" if not ok else ""
        print(f"  {mark}  {case.failure_mode:<22}{extra}")

    print("\nPASS examples — classifier should NOT detect the failure mode:")
    for case, found in pass_cases:
        ok = case.failure_mode not in found
        n_correct += ok
        mark = "✓" if ok else "✗"
        extra = f"  (also got: {sorted(found)})" if not ok else ""
        print(f"  {mark}  {case.failure_mode:<22}{extra}")

    n_total = len(cases)
    print(f"\n  Score: {n_correct}/{n_total}  ({100 * n_correct // n_total}%)")
    return n_correct


async def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GEMINI_API_KEY")

    if not openai_key and not google_key:
        sys.exit("Error: set at least one of OPENAI_API_KEY or GEMINI_API_KEY in .env")

    cases = build_test_cases()
    print(f"Built {len(cases)} test cases  ({len(FAILURE_MODES)} failure modes × 2 examples each)")

    configs: list[ModelConfig] = []
    if openai_key:
        configs.append(ModelConfig("gpt-5.4-2026-03-05", "openai", openai_key))
    if google_key:
        configs.append(ModelConfig("gemini-3.1-pro-preview", "google", google_key))

    for config in configs:
        print(f"\nRunning {len(cases)} classify() calls on {config.model} ...")
        detected = await run_cases(cases, config)
        print_results(cases, detected, config.model)


if __name__ == "__main__":
    asyncio.run(main())
