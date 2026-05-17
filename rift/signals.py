"""
Reward Variance signal from the RIFT paper.

Two modes:
- reward_variance_from_scores: pure computation from pre-existing scores (no LLM calls).
  Use this when responses have already been graded, e.g. with HealthBench data.
- reward_variance: generates responses and grades them from scratch using two model configs
  (one for generation, one for judging).
"""

import asyncio
import json
import statistics

import hopper
from hopper import CanonicalMessage, CanonicalRequest, Credentials

from .schema import ModelConfig, Rubric


def reward_variance_from_scores(scores: list[float]) -> float:
    """Compute reward variance from a list of pre-existing rubric scores."""
    if len(scores) < 2:
        raise ValueError("Need at least 2 scores to compute variance.")
    return statistics.variance(scores)


_GRADER_SYSTEM = (
    "You are an expert evaluator. Given a rubric criterion and a model response, "
    "determine whether the response satisfies the criterion."
)


def _grader_prompt(rubric: Rubric, response: str) -> str:
    return "\n\n".join([
        f"## Input Context\n{rubric.input_context}",
        f"## Rubric Criterion\n{rubric.rubric_text}",
        f"## Model Response\n{response}",
        "## Task\nReturn JSON: {\"score\": 1} if the criterion is fully met, {\"score\": 0} if not.",
    ])


async def _generate(input_context: str, model: ModelConfig) -> str:
    request = CanonicalRequest(
        model=model.model,
        provider=model.provider,
        messages=[CanonicalMessage(role="user", content=input_context)],
    )
    envelope = await hopper.complete(request, Credentials(api_key=model.api_key))
    return envelope.response.content


async def _grade(rubric: Rubric, response: str, judge: ModelConfig) -> float:
    request = CanonicalRequest(
        model=judge.model,
        provider=judge.provider,
        system=_GRADER_SYSTEM,
        messages=[CanonicalMessage(role="user", content=_grader_prompt(rubric, response))],
        extra_params=judge.json_params(),
    )
    envelope = await hopper.complete(request, Credentials(api_key=judge.api_key))
    data = json.loads(envelope.response.content)
    return float(data["score"])


async def reward_variance(
    rubric: Rubric,
    response_model: ModelConfig,
    judge: ModelConfig,
    n_responses: int = 4,
) -> float:
    """Generate n_responses to the rubric's input context, grade each, and return the variance."""
    if n_responses < 2:
        raise ValueError("Need at least 2 responses to compute variance.")
    responses = await asyncio.gather(
        *[_generate(rubric.input_context, response_model) for _ in range(n_responses)]
    )
    scores = await asyncio.gather(
        *[_grade(rubric, r, judge) for r in responses]
    )
    return reward_variance_from_scores(list(scores))
