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
    votes: list[list[str]] = field(default_factory=list)  # per-run label sets


@dataclass
class ModelConfig:
    model: str
    provider: str  # "openai" or "google"
    api_key: str

    def json_params(self) -> dict[str, Any]:
        if self.provider == "openai":
            # hopper uses the Responses API; JSON mode is set via the `text` parameter
            return {"text": {"format": {"type": "json_object"}}}
        if self.provider == "google":
            return {"response_mime_type": "application/json"}
        return {}
