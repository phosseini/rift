"""
Loaders for the five rubric datasets used in the RIFT paper (Table 5).

Each loader returns a list of Rubric objects sampled uniformly at random,
matching the paper's "stratified uniform random sampling with equal counts
per source, with a fixed random seed" procedure (Appendix A).

Dataset origins (paper Table 5):
  advancedif      expert-curated  checklist   ~88 words/rubric
  researchrubrics expert-written  checklist  ~1059 words/task-rubric → split per criterion
  wildchecklists  LLM-generated   checklist   ~111 words/rubric
  openrubrics     LLM-generated   principles  ~145 words/rubric
  autorubrics     LLM-generated   narrative    ~68 words/rubric
"""

import json
import random

from datasets import load_dataset

from ..schema import Rubric


def _parse_field(value):
    """Handle fields that HuggingFace may leave as JSON strings or already parse."""
    return json.loads(value) if isinstance(value, str) else value


def _format_conversation(turns: list[dict]) -> str:
    return "\n\n".join(
        f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}"
        for t in turns
    )


def load_advancedif(n: int, seed: int = 42) -> list[Rubric]:
    """
    facebook/AdvancedIF — expert-curated instruction-following rubrics.
    input_context : last user message from conversation_history
    rubric_text   : bulleted list of yes/no evaluation questions from prompt_metadata.rubrics
    """
    ds = load_dataset("facebook/AdvancedIF", split="train")
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    result = []
    for row in rows:
        history = _parse_field(row["conversation_history"])
        metadata = _parse_field(row["prompt_metadata"])
        user_msgs = [m["content"] for m in history if m.get("role") == "user"]
        input_context = user_msgs[-1] if user_msgs else _format_conversation(history)
        criteria = _parse_field(metadata.get("rubrics", []))
        result.append(Rubric(
            input_context=input_context,
            rubric_text="\n".join(f"- {c}" for c in criteria),
            metadata={"source": "advancedif", "benchmark": row["benchmark_name"]},
        ))
    return result


def load_researchrubrics(n: int, seed: int = 42) -> list[Rubric]:
    """
    ScaleAI/researchrubrics — expert-written rubrics for deep research tasks.

    Each task has 20-43 criteria. We treat each criterion as an independent
    Rubric so the LLMaJ evaluates one scorable item at a time — consistent
    with how RIFT failure modes like Non-Atomic and Subjective are defined
    (they apply to individual scoring criteria, not 1,000-word checklists).

    The full pool is ~2,593 criteria across 101 tasks; we sample n from that.

    input_context : research prompt for the parent task
    rubric_text   : single criterion with its axis and weight as context
    """
    ds = load_dataset("ScaleAI/researchrubrics", split="train")
    all_criteria: list[Rubric] = []
    for row in ds:
        for criterion in row["rubrics"]:
            all_criteria.append(Rubric(
                input_context=row["prompt"],
                rubric_text=(
                    f"[{criterion['axis']}, weight {criterion['weight']}] "
                    f"{criterion['criterion']}"
                ),
                metadata={
                    "source": "researchrubrics",
                    "domain": row["domain"],
                    "sample_id": row["sample_id"],
                },
            ))
    rng = random.Random(seed)
    rng.shuffle(all_criteria)
    return all_criteria[:min(n, len(all_criteria))]


def load_wildchecklists(n: int, seed: int = 42) -> list[Rubric]:
    """
    viswavi/wildchecklists — LLM-generated checklists from WildChat-1M.
    input_context : user prompt
    rubric_text   : requirements string (numbered checklist, already formatted)
    """
    ds = load_dataset("viswavi/wildchecklists", split="train")
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        Rubric(
            input_context=row["prompt"],
            rubric_text=row["requirements"],
            metadata={"source": "wildchecklists"},
        )
        for row in rows
    ]


def load_openrubrics(n: int, seed: int = 42) -> list[Rubric]:
    """
    OpenRubrics/OpenRubrics — LLM-generated principles-format rubrics.
    input_context : instruction
    rubric_text   : rubric (principles format, single string)
    """
    ds = load_dataset("OpenRubrics/OpenRubrics", split="train")
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        Rubric(
            input_context=row["instruction"],
            rubric_text=row["rubric"],
            metadata={"source": "openrubrics", "data_source": row["source"]},
        )
        for row in rows
    ]


def load_healthbench_professional(n: int | None = None) -> tuple[list[Rubric], list[Rubric]]:
    """
    openai/healthbench-professional — clinician-written rubrics for real clinical chats.

    Returns a tuple of (per_criterion, per_conversation) rubric lists.

    per_criterion  : one Rubric per rubric item — for criterion-scope failure modes
                     (subjective, non_atomic, ungrounded, misaligned_or_rigid, hackable)
    per_conversation: one Rubric per conversation with all criteria joined — for
                     rubric-scope failure modes (missing_criteria, low_signal,
                     redundant_criteria) that require seeing the full rubric set.

    input_context  : full conversation formatted as alternating turns (both lists)
    rubric_text    : single criterion with point prefix (per_criterion)
                     all criteria joined (per_conversation)
    """
    ds = load_dataset("openai/healthbench-professional", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    per_criterion: list[Rubric] = []
    per_conversation: list[Rubric] = []
    for row in ds:
        input_context = _format_conversation(row["conversation"]["messages"])
        base_meta = {
            "conversation_id": row["id"],
            "use_case": row["use_case"],
            "type": row["type"],
            "difficulty": row["difficulty"],
            "specialty": row["specialty"],
        }
        criteria_lines = []
        for item in row["rubric_items"]:
            rubric_text = f"[{item['points']:+d} pts] {item['criterion_text']}"
            per_criterion.append(Rubric(
                input_context=input_context,
                rubric_text=rubric_text,
                metadata={
                    **base_meta,
                    "eval_mode": "per_criterion",
                    "points": item["points"],
                    "criterion_text": item["criterion_text"],
                },
            ))
            criteria_lines.append(rubric_text)
        per_conversation.append(Rubric(
            input_context=input_context,
            rubric_text="\n".join(criteria_lines),
            metadata={
                **base_meta,
                "eval_mode": "per_conversation",
                "criterion_count": len(row["rubric_items"]),
            },
        ))
    return per_criterion, per_conversation


def load_autorubrics(n: int, seed: int = 42) -> list[Rubric]:
    """
    agentscope-ai/Auto-Rubric — LLM-generated narrative rubrics from preference data.
    input_context : full conversation history (formatted)
    rubric_text   : rubric criteria joined as a bulleted list
    Only rows with rubric_valid=True and at least one criterion are included.
    """
    ds = load_dataset("agentscope-ai/Auto-Rubric", split="train")
    ds = ds.filter(lambda x: x["rubric_valid"] == "True" and len(x["rubrics"]) > 0)
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    result = []
    for row in rows:
        result.append(Rubric(
            input_context=_format_conversation(row["input"]),
            rubric_text="\n".join(f"- {r}" for r in row["rubrics"]),
            metadata={"source": "autorubrics", "domain": row["domain"]},
        ))
    return result
