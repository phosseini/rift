from dataclasses import dataclass, field


@dataclass
class FailureMode:
    label: str
    category: str  # "reliability" | "content_validity" | "consequential_validity"
    scope: str     # "criterion" (per rubric item) | "rubric" (full rubric set)
    description: str
    pass_examples: list[dict[str, str]]
    fail_examples: list[dict[str, str]]


FAILURE_MODES: list[FailureMode] = [
    # ── Reliability Failures ──────────────────────────────────────────────────
    FailureMode(
        label="subjective",
        category="reliability",
        scope="criterion",
        description=(
            "Apply when the rubric uses inherently subjective evaluative terms "
            "(e.g., 'clear,' 'appropriate,' 'credible,' 'comprehensive,' 'professional,' "
            "'engaging,' 'well-written,' 'good sources') and does NOT sufficiently anchor "
            "them with objective expectations.\n\n"
            "How to determine:\n"
            "- Identify criteria dominated by inherently subjective terms.\n"
            "- Check whether the rubric provides ANY anchoring attempt: concrete checklists "
            "('includes X/Y/Z'), measurable thresholds (word count, required sections), "
            "examples/anti-examples of what qualifies vs. does not qualify, or explicit "
            "decision rules ('count as clear if it defines the term and gives one example').\n"
            "- If the rubric relies primarily on grader judgment with no meaningful anchors, apply.\n\n"
            "Do NOT apply if:\n"
            "- The rubric gives examples or non-trivial decision rules that explain the subjective "
            "term, even if the term itself is still somewhat subjective.\n"
            "- The core issue is missing expected answers or a bounded verification procedure "
            "for a groundable requirement (use Ungrounded).\n"
            "- The requirement is entirely absent (use Missing Criteria)."
        ),
        pass_examples=[{
            "input_context": "Write a professional email declining a meeting.",
            "rubric": (
                "2 pts: Includes a decline + proposes an alternative time. "
                "2 pts: Uses a greeting and sign-off. "
                "1 pt: No negative or insulting language."
            ),
        }],
        fail_examples=[{
            "input_context": "Summarize the study.",
            "rubric": "10 pts: The summary is clear and sufficiently detailed.",
        }],
    ),
    FailureMode(
        label="non_atomic",
        category="reliability",
        scope="criterion",
        description=(
            "Apply when the rubric does not provide a parseable, consistently scorable "
            "structure OR uses bundled (non-atomic) criteria that prevent consistent partial credit.\n\n"
            "Triggers (any sufficient):\n"
            "- One scored item bundles multiple independently scorable requirements with no "
            "partial-credit rule or separable sub-scores "
            "(e.g., 'clear, comprehensive, accurate, and well-cited' as a single 10-pt item).\n\n"
            "Do NOT apply when:\n"
            "- Subparts are separately scored or the rubric provides explicit level anchors "
            "(e.g., '1 point each for A/B/C' or a 0–2 scale per dimension with definitions).\n"
            "- The rubric is scorable but uses subjective language (use Subjective) or is "
            "missing requirements (use Missing Criteria)."
        ),
        pass_examples=[{
            "input_context": "Write a short answer with two supporting reasons.",
            "rubric": (
                "2 pts: Answers the question. "
                "1 pt: Reason #1 supports the answer. "
                "1 pt: Reason #2 supports the answer. "
                "1 pt: Total length <=150 words."
            ),
        }],
        fail_examples=[{
            "input_context": "Summarize the article.",
            "rubric": "10 pts: Summary is clear, comprehensive, accurate, and engaging.",
        }],
    ),
    FailureMode(
        label="ungrounded",
        category="reliability",
        scope="criterion",
        description=(
            "Apply when the rubric requires verification that is plausibly groundable or "
            "boundable, but does not provide the necessary grounding (answer keys, acceptable "
            "variants, tolerances, decision rules) OR does not bound the verification procedure "
            "(what to check, how much to check, and how to judge conflicts).\n\n"
            "How to determine (any sufficient):\n"
            "- (A) Groundable determinate tasks lack grading anchors. The task has a knowable "
            "target output given fixed inputs (extraction, classification, translation, math, "
            "SQL, code output), but the rubric provides no expected answers, acceptable variants, "
            "label mappings, tolerances, or decision rules.\n"
            "- (B) Open-world requirements lack bounded audit procedure. The rubric demands broad "
            "verification ('all facts are true,' 'links work,' 'fully original') without bounding: "
            "what to check, which sources are allowed, how to resolve conflicting evidence, "
            "and the pass/fail threshold.\n"
            "- (C) Measurement standard is unspecified but could be made checkable. The rubric "
            "requires a measurement that depends on an unspecified standard without defining "
            "a rendering standard or offering a workable proxy.\n\n"
            "Do NOT apply if:\n"
            "- The requirement is simply missing from the rubric (use Missing Criteria).\n"
            "- The main issue is subjective wording without anchors (use Subjective). The rubric "
            "provides a representative list of examples to demonstrate expected content."
        ),
        pass_examples=[{
            "input_context": "Extract all email addresses from the text.",
            "rubric": (
                "1 pt per correct email address; accepted forms include plus-addressing. "
                "Gold list of emails: a@x.com, b.y@z.org. Deduct 1 pt per missing email."
            ),
        }],
        fail_examples=[{
            "input_context": "Compute the correct totals for these 30 invoices.",
            "rubric": "10 pts: Totals are correct.",
        }],
    ),

    # ── Content Validity Failures ─────────────────────────────────────────────
    FailureMode(
        label="misaligned_or_rigid",
        category="content_validity",
        scope="criterion",
        description=(
            "Apply when the rubric (a) grades the wrong objective for the prompt or embeds "
            "incorrect assumptions, OR (b) imposes unnecessarily strict or narrow requirements "
            "not asked for by the prompt, predictably penalizing prompt-faithful high-quality answers.\n\n"
            "How to determine (any applies):\n"
            "- Wrong task / shifted objective: makes non-requested deliverables mandatory for points.\n"
            "- Incorrect embedded assumptions: assumes a context not in the prompt (jurisdiction, "
            "audience, tools, constraints) and scores accordingly.\n"
            "- Penalizes good practice: scores down reasonable caveats or uncertainty expressions "
            "when the prompt does not forbid them.\n"
            "- Arbitrary brittleness/over-constraint: mandates a specific tool, library, method, "
            "structure, or formatting when multiple reasonable alternatives satisfy the prompt.\n\n"
            "Do NOT apply when:\n"
            "- The prompt itself imposes the strictness (e.g., exact JSON keys, or a direct "
            "instruction from the user in a chat conversation).\n"
            "- The requirement is missing entirely (use Missing Criteria).\n"
            "- The main problem is rubric-level proxy gaming (use Hackable)."
        ),
        pass_examples=[{
            "input_context": "Write Python code to parse CSV.",
            "rubric": (
                "5 pts: Correct parsing. "
                "2 pts: Handles quoted commas. "
                "2 pts: Includes brief usage example. "
                "(Does not mandate pandas vs csv module.)"
            ),
        }],
        fail_examples=[{
            "input_context": "Write a haiku about winter.",
            "rubric": (
                "5 pts: Includes at least 5 academic citations. "
                "5 pts: Uses APA format reference list."
            ),
        }],
    ),
    FailureMode(
        label="missing_criteria",
        category="content_validity",
        scope="rubric",
        description=(
            "Apply when the prompt implies at least one checkable must-have requirement, "
            "but the rubric provides no criterion that allows a grader to evaluate that "
            "requirement at all.\n\n"
            "How to determine:\n"
            "1. List the prompt's core requirements: required deliverables/components, "
            "must/must-not constraints, required format/ordering/sections, and genre-critical "
            "qualities the prompt clearly expects "
            "(e.g., functional correctness for code; \"two sentences\"; \"valid JSON\"; "
            "\"include 10 items\"; \"chronological order\").\n"
            "2. For each requirement, check whether ANY rubric criterion covers it.\n"
            "3. If one or more requirements have no corresponding criterion, apply.\n\n"
            "Do NOT apply if:\n"
            "- The rubric mentions the requirement but is vague or subjective (use Subjective).\n"
            "- The rubric mentions the requirement but cannot be graded consistently due to "
            "missing tolerances or bounded audit steps (use Ungrounded).\n"
            "- The rubric grades a different task or adds arbitrary constraints "
            "(use Misaligned or Rigid)."
        ),
        pass_examples=[{
            "input_context": "Return ONLY valid JSON with keys: name (string) and age (integer).",
            "rubric": (
                "3 pts: Output parses as JSON. "
                "2 pts: Contains exactly keys name and age. "
                "1 pt: name is a string; age is an integer. "
                "1 pt: No surrounding commentary."
            ),
        }],
        fail_examples=[{
            "input_context": "Write a 200-word email and include a subject line.",
            "rubric": "10 pts: Tone is professional. 5 pts: Grammar and spelling are correct.",
        }],
    ),

    # ── Consequential Validity Failures ──────────────────────────────────────
    FailureMode(
        label="hackable",
        category="consequential_validity",
        scope="criterion",
        description=(
            "Apply when the rubric is gameable at the rubric level: a responder could easily "
            "achieve a top score by inflating proxy metrics (length, number of bullets/sections/"
            "items/citations/examples, repeated keywords) without materially improving correctness, "
            "relevance, or fulfillment of the prompt—and the rubric lacks strong quality gates "
            "that tie points to substantive, prompt-aligned success.\n\n"
            "Core question (required): Could I easily achieve full marks while still not "
            "satisfying the prompt requirements or producing a low-quality response?\n\n"
            "How to determine (any sufficient):\n"
            "- Most points come from 'more' counts (≥N tips/citations/examples/pros/cons) while "
            "relevance, non-duplication, and correctness are weakly specified or absent.\n"
            "- Rewards merely asserting attributes without requiring evidence or linkage to the task.\n"
            "- Counting proxies dominate while key prompt requirements have only weak gates.\n\n"
            "Do NOT apply when:\n"
            "- Quantity minimums are paired with robust quality controls that make padding ineffective.\n"
            "- The main issue is that the rubric is generic and does not discriminate at all "
            "(use Low Signal).\n"
            "- The main issue is a specific criterion that shifts the task or overconstrains "
            "acceptable answers (use Misaligned or Rigid)."
        ),
        pass_examples=[{
            "input_context": "Provide 5 study tips.",
            "rubric": (
                "1 pt each for 5 tips that are (a) non-duplicative and "
                "(b) each includes a concrete example of how to apply it."
            ),
        }],
        fail_examples=[{
            "input_context": "Provide a recommendation.",
            "rubric": (
                "5 pts: At least 10 pros. "
                "5 pts: At least 10 cons. "
                "(No check for relevance or duplication.)"
            ),
        }],
    ),
    FailureMode(
        label="low_signal",
        category="consequential_validity",
        scope="rubric",
        description=(
            "Apply when the rubric as a whole does not discriminate candidate responses well "
            "for this prompt—it would give nearly equivalent scores to many substantively "
            "different-quality responses—because the criteria are all generic, conditionally "
            "irrelevant, or too easy.\n\n"
            "How to determine (rubric-level discrimination test):\n"
            "- Imagine 3–5 candidate responses ranging from weak to excellent.\n"
            "- Ask: Would the rubric's criteria/weights produce nearly equivalent scores across "
            "them based on prompt-relevant success?\n"
            "- If most criteria are low-signal (e.g., 'helpful,' 'nice formatting,' 'completed "
            "the task') and there are few or no strong quality gates, apply.\n\n"
            "Common signals:\n"
            "- Most points are allocated to generic writing quality or tone not central to this prompt.\n"
            "- Criteria are trivially satisfied by any minimally on-task response.\n"
            "- The rubric could be pasted into many unrelated tasks with little or no change.\n\n"
            "Do NOT apply when:\n"
            "- The rubric is missing prompt-imposed must-haves (use Missing Criteria).\n"
            "- The rubric imposes the wrong constraints or assumptions (use Misaligned or Rigid).\n"
            "- The rubric is gameable specifically via quantity or proxies (use Hackable)."
        ),
        pass_examples=[{
            "input_context": "Return JSON only with required keys.",
            "rubric": "6 pts: Valid JSON with required keys. 4 pts: No extra text outside JSON.",
        }],
        fail_examples=[{
            "input_context": "Return only a SQL query.",
            "rubric": "5 pts: Response is helpful. 5 pts: Uses appropriate tone.",
        }],
    ),
    FailureMode(
        label="redundant_criteria",
        category="consequential_validity",
        scope="rubric",
        description=(
            "Apply when two or more rubric criteria substantially evaluate the same underlying "
            "requirement such that the same behavior is rewarded or penalized multiple times.\n\n"
            "How to determine:\n"
            "- Additional-signal test (primary): If these criteria are separate, do you get "
            "genuinely different evaluation signal, or are you just re-awarding the same property?\n"
            "- Remove-one test: If removing a criterion would not meaningfully change what is "
            "evaluated (only point allocation), it is redundant.\n"
            "- Includes near-duplicates and cases where one criterion fully subsumes another.\n\n"
            "Do NOT apply for mere dependencies:\n"
            "- Do NOT apply just because criteria are related or one tends to enable another.\n"
            "- If Criterion B is a prerequisite for Criterion A but still measures a distinct "
            "dimension (e.g., 'valid JSON' and 'has required keys'; 'code compiles' and 'passes "
            "tests'), that is NOT redundancy.\n"
            "- Do NOT apply when criteria are related but clearly distinct "
            "(factual accuracy vs. clarity; format compliance vs. correctness; presence of "
            "citations vs. whether citations support claims)."
        ),
        pass_examples=[{
            "input_context": "Write a research summary with citations.",
            "rubric": (
                "3 pts: Claims are supported by citations. "
                "2 pts: Writing is well-organized. "
                "2 pts: Evidence is grounded."
            ),
        }],
        fail_examples=[{
            "input_context": "Essay rubric.",
            "rubric": (
                "5 pts: Clear writing. "
                "5 pts: Clarity of prose. "
                "5 pts: Writing is easy to understand."
            ),
        }],
    ),
]

FAILURE_MODE_LABELS: set[str] = {fm.label for fm in FAILURE_MODES}
