import asyncio
import json
import math

import hopper
from hopper import CanonicalMessage, CanonicalRequest, Credentials

from .prompts import LLMAJ_SYSTEM, build_llmaj_prompt
from .schema import DiagnosticResult, FailureModeLabel, ModelConfig, Rubric
from .taxonomy import FAILURE_MODES, FailureMode


def _normalize_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_labels(content: str, valid_labels: set[str]) -> list[FailureModeLabel]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else data.get("suggested_labels", [])
    labels = []
    for item in items:
        normalized = _normalize_label(item.get("label", ""))
        if normalized in valid_labels:
            labels.append(FailureModeLabel(
                label=normalized,
                justification=item["justification"],
                quote=item["quote"],
            ))
    return labels


async def _run_once(
    rubric: Rubric,
    config: ModelConfig,
    failure_modes: list[FailureMode],
) -> list[FailureModeLabel]:
    valid_labels = {fm.label for fm in failure_modes}
    request = CanonicalRequest(
        model=config.model,
        provider=config.provider,
        system=LLMAJ_SYSTEM,
        messages=[CanonicalMessage(role="user", content=build_llmaj_prompt(rubric, failure_modes))],
        extra_params=config.json_params(),
    )
    envelope = await hopper.complete(request, Credentials(api_key=config.api_key))
    return _parse_labels(envelope.response.content, valid_labels)


async def classify(
    rubric: Rubric,
    config: ModelConfig,
    failure_modes: list[FailureMode] | None = None,
    n_votes: int = 1,
) -> DiagnosticResult:
    fms = failure_modes if failure_modes is not None else FAILURE_MODES
    if n_votes == 1:
        labels = await _run_once(rubric, config, fms)
        return DiagnosticResult(rubric=rubric, labels=labels, model=config.model, n_votes=1)

    runs = await asyncio.gather(*[_run_once(rubric, config, fms) for _ in range(n_votes)])
    threshold = math.ceil(n_votes / 2)

    vote_counts: dict[str, list[FailureModeLabel]] = {}
    for run_labels in runs:
        seen = set()
        for label in run_labels:
            if label.label not in seen:
                vote_counts.setdefault(label.label, []).append(label)
                seen.add(label.label)

    majority_labels = [
        instances[0]
        for instances in vote_counts.values()
        if len(instances) >= threshold
    ]
    per_run_votes = [sorted(lbl.label for lbl in run_labels) for run_labels in runs]
    return DiagnosticResult(
        rubric=rubric,
        labels=majority_labels,
        model=config.model,
        n_votes=n_votes,
        votes=per_run_votes,
    )
