from .schema import Rubric
from .taxonomy import FAILURE_MODES, FailureMode


LLMAJ_SYSTEM = (
    "You are an expert at evaluating rubric quality. Analyze the following rubric "
    "against the failure mode taxonomy and identify any issues. The rubric is designed "
    "to evaluate the quality of an AI model's response to a given prompt."
)

_RESPONSE_SCHEMA = """\
Return your response as JSON with this schema:

{
  "suggested_labels": [
    {
      "label": "<failure mode label exactly as shown in the taxonomy headers, e.g. subjective, non_atomic>",
      "justification": "<why this failure mode applies to this specific rubric>",
      "quote": "<verbatim quote from the rubric that exhibits the issue>"
    }
  ]
}

If no failure modes apply, return {"suggested_labels": []}.\
"""


def _format_failure_mode(fm: FailureMode) -> str:
    lines = [f"### {fm.label}", f"Description: {fm.description}", ""]
    lines.append("**Pass Examples** (rubric does NOT exhibit this failure mode):")
    for ex in fm.pass_examples:
        lines.append(f"- Input: {ex['input_context'][:150]}...")
        lines.append(f"  Rubric: {ex['rubric'][:200]}...")
    lines.append("")
    lines.append("**Fail Examples** (rubric DOES exhibit this failure mode):")
    for ex in fm.fail_examples:
        lines.append(f"- Input: {ex['input_context'][:150]}...")
        lines.append(f"  Rubric: {ex['rubric'][:200]}...")
    return "\n".join(lines)


def build_llmaj_prompt(rubric: Rubric, failure_modes: list[FailureMode] | None = None) -> str:
    fms = failure_modes if failure_modes is not None else FAILURE_MODES
    taxonomy_section = "\n\n".join(_format_failure_mode(fm) for fm in fms)
    return "\n\n".join([
        "## Failure Mode Taxonomy",
        taxonomy_section,
        f"## Input Context\n{rubric.input_context}",
        f"## Rubric to Evaluate\n{rubric.rubric_text}",
        f"## Task\nIdentify which failure modes from the taxonomy apply to this rubric (if any).\n\n{_RESPONSE_SCHEMA}",
    ])
