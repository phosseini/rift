from dataclasses import dataclass, field
from typing import Any


@dataclass
class Rubric:
    rubric_text: str
    input_context: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureModeLabel:
    label: str
    justification: str
    quote: str


@dataclass
class DiagnosticResult:
    rubric: Rubric
    labels: list[FailureModeLabel]       # majority-voted final labels
    model: str
    n_votes: int = 1
    votes: list[list[FailureModeLabel]] = field(default_factory=list)  # per-run label sets
    # Token usage summed over all runs: {"calls", "input_tokens", "output_tokens", "cached_input_tokens"}
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelConfig:
    model: str
    provider: str  # "openai", "google", or "portkey"
    api_key: str
    # ── optional routing overrides ────────────────────────────────────────
    # `model` is the display name recorded in results (judge_model). When a
    # gateway needs a different identifier on the wire (e.g. Portkey's
    # "@openai/gpt-5.4-2026-03-05"), put it in `api_model`.
    api_model: str | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Ask the provider for JSON output. Disable if the routed model rejects it;
    # the response parser tolerates fenced/plain JSON either way.
    json_mode: bool = True

    @property
    def wire_model(self) -> str:
        return self.api_model or self.model

    def json_params(self) -> dict[str, Any]:
        if not self.json_mode:
            return {}
        if self.provider == "openai":
            # hopper uses the Responses API; JSON mode is set via the `text` parameter
            return {"text": {"format": {"type": "json_object"}}}
        if self.provider == "google":
            return {"response_mime_type": "application/json"}
        return {}
