import asyncio
import json
import math
import re

import hopper
from hopper import CanonicalMessage, CanonicalRequest, Credentials

from .prompts import LLMAJ_SYSTEM, build_llmaj_prompt
from .schema import DiagnosticResult, FailureModeLabel, ModelConfig, Rubric
from .taxonomy import FAILURE_MODES, FailureMode

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _normalize_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _extract_json(content: str):
    """Parse a JSON object from model output, tolerating ```json fences and
    leading/trailing prose (models routed without native JSON mode do this)."""
    text = content.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


class JudgeOutputError(RuntimeError):
    """Judge reply was truncated or not parseable JSON. Raised (not swallowed) so the
    caller retries and, if it keeps failing, records an error instead of an empty
    label set — otherwise a cut-off reply is indistinguishable from "no failure modes"."""


def _parse_labels(content: str, valid_labels: set[str]) -> list[FailureModeLabel]:
    data = _extract_json(content)
    if data is None:
        raise JudgeOutputError(f"unparseable judge output: {content[:200]!r}")
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


async def _complete_portkey(config: ModelConfig, system: str, user: str) -> tuple[str, str, dict]:
    """Call an OpenAI-compatible gateway (Portkey) via Chat Completions.

    Hopper's OpenAI adapter speaks the Responses API, which gateways only
    proxy for OpenAI-hosted models. Chat Completions is the lingua franca
    Portkey translates for every upstream provider (OpenAI, Anthropic, Google,
    Bedrock, OpenRouter, ...), so the gateway route uses it directly.
    """
    import openai
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=config.extra_headers or None,
    )
    kwargs: dict = {
        "model": config.wire_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        # Some upstreams behind the gateway reject response_format; the prompt
        # already demands JSON and the parser tolerates fences, so retry bare.
        if "response_format" in kwargs and "response_format" in str(e):
            kwargs.pop("response_format")
            resp = await client.chat.completions.create(**kwargs)
        else:
            raise
    choice = resp.choices[0]
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None) if u else None
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cached_input_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
    }
    return choice.message.content or "", choice.finish_reason or "stop", usage


def _check_complete(finish_reason: str) -> None:
    # Any finish reason other than a normal stop means the JSON may be cut off.
    if finish_reason not in ("stop", "completed", "end_turn"):
        raise JudgeOutputError(f"judge output incomplete (finish_reason={finish_reason})")


async def _run_once(
    rubric: Rubric,
    config: ModelConfig,
    failure_modes: list[FailureMode],
) -> tuple[list[FailureModeLabel], dict]:
    """One judge call. Returns (labels, usage) where usage counts this call's tokens."""
    valid_labels = {fm.label for fm in failure_modes}
    user_prompt = build_llmaj_prompt(rubric, failure_modes)

    if config.provider == "portkey":
        content, finish_reason, usage = await _complete_portkey(config, LLMAJ_SYSTEM, user_prompt)
        _check_complete(finish_reason)
        return _parse_labels(content, valid_labels), usage

    request = CanonicalRequest(
        model=config.wire_model,
        provider=config.provider,
        system=LLMAJ_SYSTEM,
        messages=[CanonicalMessage(role="user", content=user_prompt)],
        extra_params=config.json_params(),
    )
    credentials = Credentials(api_key=config.api_key, base_url=config.base_url)
    envelope = await hopper.complete(request, credentials)
    _check_complete(envelope.response.finish_reason)
    u = envelope.usage
    usage = {
        "input_tokens": u.input_tokens if u else 0,
        "output_tokens": u.output_tokens if u else 0,
        "cached_input_tokens": 0,  # hopper does not surface cache reads
    }
    return _parse_labels(envelope.response.content, valid_labels), usage


async def classify(
    rubric: Rubric,
    config: ModelConfig,
    failure_modes: list[FailureMode] | None = None,
    n_votes: int = 1,
) -> DiagnosticResult:
    fms = failure_modes if failure_modes is not None else FAILURE_MODES
    if n_votes == 1:
        labels, usage = await _run_once(rubric, config, fms)
        return DiagnosticResult(rubric=rubric, labels=labels, model=config.model, n_votes=1, votes=[labels],
                                usage={"calls": 1, **usage})

    outcomes = await asyncio.gather(*[_run_once(rubric, config, fms) for _ in range(n_votes)])
    runs = [labels for labels, _ in outcomes]
    usage = {"calls": len(outcomes)}
    for _, u in outcomes:
        for k, v in u.items():
            usage[k] = usage.get(k, 0) + v
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
    per_run_votes = [
        sorted(run_labels, key=lambda lbl: lbl.label)
        for run_labels in runs
    ]
    return DiagnosticResult(
        rubric=rubric,
        labels=majority_labels,
        model=config.model,
        n_votes=n_votes,
        votes=per_run_votes,
        usage=usage,
    )
