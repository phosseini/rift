"""
Shared experiment pipeline for every rubric dataset.

A dataset script only has to (1) load its rubrics into a `Dataset` and
(2) say which metadata fields the console report should break down by.
Everything else — config/rift.yaml handling, the concurrent classification loop
with retries, incremental JSONL writing, result caching, the prevalence
tables and flagged examples — lives here and is identical across datasets.

    from rift.experiment import Dataset, add_common_args, run_and_analyze

    ds = Dataset(
        name="my_dataset", results_prefix="mydata",
        per_criterion=per_criterion_rubrics, per_rubric=per_rubric_rubrics,
        group_fields=["domain", "sign"],
    )
    asyncio.run(run_and_analyze(ds, args))

Result records have a stable schema across datasets:

    dataset, judge_model, judge_provider, eval_mode, <all rubric metadata>,
    rubric_text, labels, votes, included_failure_modes, n_votes, [error]

Evaluation strategies (config/rift.yaml evaluation.strategy or --eval-strategy):
    joined  — every rubric evaluated once with all enabled failure modes (eval_mode="joined")
    scoped  — criterion-scope modes per criterion (eval_mode="per_criterion"),
              rubric-scope modes on the full rubric (eval_mode=Dataset.rubric_eval_mode)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tqdm.asyncio import tqdm

from . import config as rift_config
from .classifier import classify
from .judges import default_judge, describe, resolve_judge
from .schema import ModelConfig, Rubric
from .taxonomy import FAILURE_MODES, FailureMode

RESULTS_DIR = rift_config.REPO_ROOT / "results"  # repo-root results/, whatever the working directory

# ── config ────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    """Evaluation settings from config/rift.yaml (+ overlay): {eval_strategy, votes, failure_modes}."""
    ev = rift_config.section("evaluation")
    return {"eval_strategy": ev.get("strategy", "scoped"), "votes": ev.get("votes", 1),
            "failure_modes": ev.get("failure_modes") or {fm.label: fm.scope for fm in FAILURE_MODES}}


def resolve_failure_modes(fm_cfg: dict[str, str]) -> tuple[list[FailureMode], list[FailureMode]]:
    """(criterion_fms, rubric_fms) from config; config scope overrides taxonomy default."""
    criterion_fms = [fm for fm in FAILURE_MODES if fm_cfg.get(fm.label) == "criterion"]
    rubric_fms = [fm for fm in FAILURE_MODES if fm_cfg.get(fm.label) == "rubric"]
    return criterion_fms, rubric_fms

# ── dataset description ───────────────────────────────────────────────────────


@dataclass
class Dataset:
    name: str                              # value of the `dataset` field in every record
    results_prefix: str                    # results/<prefix>_<timestamp>.jsonl
    per_criterion: list[Rubric]            # one Rubric per criterion (criterion-scope modes)
    per_rubric: list[Rubric]               # one Rubric per full rubric (rubric-scope modes / joined)
    rubric_eval_mode: str = "per_rubric"   # eval_mode label for the rubric-level scoped pass
    group_fields: list[str] = field(default_factory=list)  # metadata fields to break prevalence down by
    cache_match: dict = field(default_factory=dict)         # first-record fields a cached file must match
    include_input_context: bool = False    # also store input_context in records
    unit: str = "rubrics"                  # noun for log lines
    # Breakdown fields computed from a record instead of stored in it, e.g. a role
    # derived from a slot number for result files written before the field existed.
    derived_fields: dict[str, Callable[[dict], object]] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{len(self.per_criterion)} criteria across {len(self.per_rubric)} {self.unit}"

# ── classification runner ─────────────────────────────────────────────────────


async def run_classify(
    rubrics: list[Rubric],
    config: ModelConfig,
    failure_modes: list[FailureMode],
    concurrency: int,
    out_path: Path,
    dataset_name: str,
    eval_mode: str,
    append: bool = False,
    n_votes: int = 1,
    include_input_context: bool = False,
    timeout: int = 180,
) -> list[dict]:
    """Classify every rubric concurrently, append records to out_path as they finish."""
    semaphore = asyncio.Semaphore(concurrency)
    fm_labels = sorted(fm.label for fm in failure_modes)

    async def classify_one(rubric: Rubric):
        async with semaphore:
            last_error = None
            for attempt in range(3):
                try:
                    result = await asyncio.wait_for(
                        classify(rubric, config, failure_modes=failure_modes, n_votes=n_votes),
                        timeout=timeout,
                    )
                    votes = [
                        [{"label": l.label, "justification": l.justification, "quote": l.quote} for l in run]
                        for run in result.votes
                    ]
                    return rubric, {l.label for l in result.labels}, votes, result.usage, None
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** attempt)
            return rubric, set(), [], {}, last_error

    total = len(rubrics)
    print(f"  [{eval_mode}] Classifying {total} rubrics with {config.model} (concurrency={concurrency}) ...")
    tasks = [asyncio.create_task(classify_one(r)) for r in rubrics]
    records, failed = [], 0
    with open(out_path, "a" if append else "w") as f, tqdm(total=total, unit="rubric", dynamic_ncols=True) as bar:
        for coro in asyncio.as_completed(tasks):
            rubric, labels, votes, usage, error = await coro
            record = {
                "dataset": dataset_name,
                "judge_model": config.model,
                "judge_provider": config.provider,
                **rubric.metadata,
                "eval_mode": eval_mode,
            }
            if include_input_context:
                record["input_context"] = rubric.input_context
            record.update({
                "rubric_text": rubric.rubric_text,
                "labels": sorted(labels),
                "votes": votes,
                "included_failure_modes": fm_labels,
                "n_votes": n_votes,
                "usage": usage,
            })
            if error:
                record["error"] = error
                failed += 1
            f.write(json.dumps(record) + "\n")
            f.flush()
            records.append(record)
            bar.update(1)
    if failed:
        print(f"  [warn] {failed}/{total} calls failed — error field set in JSONL records.")
    return records

# ── results & caching ─────────────────────────────────────────────────────────


def result_path(prefix: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"{prefix}_{ts}.jsonl"


def expected_modes(strategy: str, rubric_eval_mode: str) -> set[str]:
    return {"joined"} if strategy == "joined" else {"per_criterion", rubric_eval_mode}


def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_cached(prefix: str, judge: str, required_modes: set[str], match: dict | None = None,
                expected_counts: dict[str, int] | None = None) -> list[dict] | None:
    """Most recent results/<prefix>_*.jsonl for this judge covering required_modes and matching `match`.

    `expected_counts` (eval_mode -> record count) guards against a partial or --n
    smoke-test file being mistaken for a full run."""
    match = {"judge_model": judge, **(match or {})}
    for path in sorted(RESULTS_DIR.glob(f"{prefix}_*.jsonl"), reverse=True):
        records = load_results(path)
        if not records:
            continue
        first = records[0]
        if any(first.get(k) != v for k, v in match.items()):
            continue
        counts = Counter(r.get("eval_mode") for r in records)
        if not required_modes.issubset(counts):
            continue
        if expected_counts and any(counts.get(m, 0) < n for m, n in expected_counts.items()):
            continue
        print(f"  Cached: {path.name}  judge_model={judge}  n={len(records)}")
        return records
    return None

# ── orchestration ─────────────────────────────────────────────────────────────


# ── run manifest & cost ───────────────────────────────────────────────────────


def load_pricing() -> dict:
    """config/rift.yaml `pricing`: {model_key: {input_per_1m, output_per_1m, cached_input_per_1m}}."""
    return rift_config.section("pricing")


def find_rates(pricing: dict, config: ModelConfig) -> tuple[dict | None, str | None]:
    """Look up rates by judge name, wire model, then wire model without its Portkey slug."""
    candidates = [config.model, config.wire_model]
    if "/" in config.wire_model:
        candidates.append(config.wire_model.split("/", 1)[1])
        candidates.append(config.wire_model.rsplit("/", 1)[1])
    for key in candidates:
        if key in pricing:
            return pricing[key], key
    return None, None


def compute_cost(usage: dict, rates: dict | None) -> dict:
    """USD cost from token totals; cached input tokens are billed at the cached rate when one is given."""
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    if not rates or rates.get("input_per_1m") is None or rates.get("output_per_1m") is None:
        return {"usd": None, "complete": False, "note": "no rates for this model under `pricing` in config/rift.yaml"}
    cached_rate = rates.get("cached_input_per_1m")
    uncached = inp - cached if cached_rate is not None else inp
    usd = (uncached * rates["input_per_1m"] + out * rates["output_per_1m"]
           + (cached * cached_rate if cached_rate is not None else 0)) / 1_000_000
    return {"usd": round(usd, 4), "complete": True,
            "breakdown_usd": {"input": round(uncached * rates["input_per_1m"] / 1e6, 4),
                              "cached_input": round((cached * cached_rate / 1e6) if cached_rate is not None else 0, 4),
                              "output": round(out * rates["output_per_1m"] / 1e6, 4)}}


def _git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


class RunManifest:
    """results/<run_id>.meta.json — every setting needed to reproduce or account for a run.

    Written once before the first call (so an interrupted run still leaves its
    settings behind) and rewritten at the end with totals, token usage and cost."""

    def __init__(self, path: Path, ds: Dataset, config: ModelConfig, strategy: str, n_votes: int,
                 concurrency: int, cli_args: dict | None, fm_cfg: dict):
        self.path = path.with_suffix(".meta.json")
        self.started = time.time()
        self.data = {
            "run_id": path.stem,
            "results_file": path.name,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None,
            "duration_s": None,
            "status": "running",
            "dataset": {"name": ds.name, "n_criteria": len(ds.per_criterion), "n_rubrics": len(ds.per_rubric),
                        "rubric_eval_mode": ds.rubric_eval_mode, "cache_match": ds.cache_match},
            "judge": {"name": config.model, "provider": config.provider, "wire_model": config.wire_model,
                      "base_url": config.base_url, "json_mode": config.json_mode,
                      "extra_headers": sorted(config.extra_headers)},
            "settings": {"eval_strategy": strategy, "n_votes": n_votes, "concurrency": concurrency,
                         "config_overlay": rift_config.load().get("_overlay"),
                         "failure_modes": fm_cfg},
            "cli_args": _jsonable(cli_args or {}),
            "command": " ".join(sys.argv),
            "git": _git_info(),
            "totals": None,
            "cost": None,
        }
        self.write()

    def write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def finish(self, records: list[dict], config: ModelConfig) -> dict:
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        for r in records:
            for k in usage:
                usage[k] += (r.get("usage") or {}).get(k, 0)
        rates, key = find_rates(load_pricing(), config)
        cost = compute_cost(usage, rates)
        cost["rates"] = rates
        cost["pricing_key"] = key
        self.data.update({
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.started, 1),
            "status": "complete",
            "totals": {"records": len(records), "errors": sum(1 for r in records if r.get("error")),
                       "by_eval_mode": dict(Counter(r.get("eval_mode") for r in records)), **usage},
            "cost": cost,
        })
        self.write()
        return self.data


def print_manifest_summary(m: dict) -> None:
    t, c = m["totals"], m["cost"]
    cost_str = f"${c['usd']:.2f}" if c.get("usd") is not None else f"unknown ({c.get('note', '')})"
    print(f"  Run {m['run_id']}: {t['calls']} calls, {t['input_tokens']:,} in / {t['output_tokens']:,} out tokens"
          f" ({t['cached_input_tokens']:,} cached), {t['errors']} errors, {m['duration_s']}s, cost {cost_str}")
    print(f"  Manifest: results/{Path(m['results_file']).stem}.meta.json")


async def run_dataset(
    ds: Dataset,
    judge: str,
    strategy: str,
    concurrency: int,
    n_votes: int = 1,
    no_cache: bool = False,
    cli_args: dict | None = None,
) -> list[dict]:
    """Run (or load from cache) one judge on one dataset. Credentials are only resolved when a run is needed."""
    cfg = load_config()
    fm_cfg = cfg.get("failure_modes", {fm.label: fm.scope for fm in FAILURE_MODES})
    all_fms = [fm for fm in FAILURE_MODES if fm.label in fm_cfg]
    criterion_fms, rubric_fms = resolve_failure_modes(fm_cfg)

    if strategy == "joined":
        expected = {"joined": len(ds.per_rubric)}
    else:
        expected = {"per_criterion": len(ds.per_criterion), ds.rubric_eval_mode: len(ds.per_rubric)}

    if not no_cache:
        cached = load_cached(ds.results_prefix, judge, set(expected), ds.cache_match, expected)
        if cached is not None:
            print(f"\nLoaded cached results for {judge}  ({len(cached)} records)")
            return cached

    config = resolve_judge(judge)
    print(f"Judge: {describe(config)}")
    path = result_path(ds.results_prefix)
    if config.provider == "portkey":
        # Tag every request so the run can be found (and its cost cross-checked) in Portkey's logs.
        config.extra_headers["x-portkey-metadata"] = json.dumps({"run_id": path.stem, "dataset": ds.name, "app": "rift"})
    manifest = RunManifest(path, ds, config, strategy, n_votes, concurrency, cli_args, fm_cfg)

    records: list[dict] = []
    common = dict(config=config, concurrency=concurrency, out_path=path, dataset_name=ds.name,
                  n_votes=n_votes, include_input_context=ds.include_input_context)

    if strategy == "joined" and all_fms:
        records += await run_classify(ds.per_rubric, failure_modes=all_fms, eval_mode="joined", **common)
    elif strategy == "scoped":
        if criterion_fms:
            records += await run_classify(ds.per_criterion, failure_modes=criterion_fms, eval_mode="per_criterion", **common)
        if rubric_fms:
            records += await run_classify(ds.per_rubric, failure_modes=rubric_fms, eval_mode=ds.rubric_eval_mode,
                                          append=bool(records), **common)
    print(f"  Results saved to {path}")
    print_manifest_summary(manifest.finish(records, config))
    return records


def print_strategy(strategy: str) -> None:
    cfg = load_config()
    fm_cfg = cfg.get("failure_modes", {fm.label: fm.scope for fm in FAILURE_MODES})
    criterion_fms, rubric_fms = resolve_failure_modes(fm_cfg)
    print(f"Strategy: {strategy}")
    print(f"Enabled failure modes: {[fm.label for fm in FAILURE_MODES if fm.label in fm_cfg]}")
    if strategy == "scoped":
        print(f"  criterion-scope: {[fm.label for fm in criterion_fms]}")
        print(f"  rubric-scope:    {[fm.label for fm in rubric_fms]}")


def add_common_args(parser: argparse.ArgumentParser, default_concurrency: int = 4) -> None:
    parser.add_argument("--config", default=None, metavar="YAML",
                        help=f"Overlay YAML merged over config/rift.yaml (also ${rift_config.OVERLAY_ENV})")
    parser.add_argument("--judge", nargs="+", default=None, metavar="MODEL",
                        help="Judge model(s): registered id, @portkey/catalog/address, or portkey:<model> (default: judges.default in config)")
    parser.add_argument("--concurrency", type=int, default=default_concurrency, help="Max simultaneous API calls")
    parser.add_argument("--eval-strategy", choices=["joined", "scoped"], default=None,
                        help="Override evaluation.strategy from config/rift.yaml")
    parser.add_argument("--votes", type=int, default=None, help="Judge runs per rubric; majority vote when >1 (default: evaluation.votes in config)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-run even if cached results exist")


async def run_and_analyze(ds: Dataset, args: argparse.Namespace) -> dict[str, list[dict]]:
    """Standard main loop: for each judge, run or load, then print the console analysis."""
    if getattr(args, "config", None):
        rift_config.set_overlay(args.config)
    cfg = load_config()
    strategy = args.eval_strategy or cfg["eval_strategy"]
    if args.votes is None:
        args.votes = cfg["votes"]
    if not args.judge:
        args.judge = [default_judge()]
    if "_overlay" in rift_config.load():
        print(f"Config overlay: {rift_config.load()['_overlay']}")
    print_strategy(strategy)
    print(f"Dataset: {ds.name}  ({ds.describe()})")
    out = {}
    for judge in args.judge:
        records = await run_dataset(ds, judge, strategy, args.concurrency, args.votes, args.no_cache,
                                    cli_args=vars(args))
        analyze(records, judge, ds.group_fields, ds.derived_fields)
        out[judge] = records
    return out

# ── console analysis ──────────────────────────────────────────────────────────

# Fields computed from a record rather than stored in it.
DERIVED_FIELDS = {
    "sign": lambda r: ("positive (+pts)" if r.get("points", 0) > 0 else "negative (−pts)" if r.get("points", 0) < 0 else None),
}


def field_value(record: dict, field_name: str, derived: dict | None = None):
    fn = (derived or {}).get(field_name) or DERIVED_FIELDS.get(field_name)
    if fn is not None:
        return fn(record)
    return record.get(field_name)


def prevalence(records: list[dict], label: str) -> float:
    return 100 * sum(1 for r in records if label in r["labels"]) / len(records) if records else 0.0


def print_table(title: str, groups: dict[str, list[dict]], labels: list[str], min_width: int = 16) -> None:
    groups = {str(k): v for k, v in groups.items() if v}
    if not groups:
        return
    col_w = max(min_width, max(len(k) for k in groups) + 2)
    header = f"  {'':26}" + "".join(f"{k:>{col_w}}" for k in groups)
    width = max(len(header), 68)
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}\n{header}\n  " + "-" * (26 + col_w * len(groups)))
    for label in labels:
        row = f"  {label:<26}"
        for recs in groups.values():
            pct = prevalence(recs, label)
            row += f"{pct:>{col_w - 1}.0f}%" if pct > 0 else f"{'—':>{col_w}}"
        print(row)
    print(f"  {'':26}" + "".join(f"{len(v):>{col_w}}" for v in groups.values()))
    print(f"  {'':26}" + "".join(f"{'(n)':>{col_w}}" for _ in groups))


def print_flagged_examples(records: list[dict], labels: list[str], group_fields: list[str], derived: dict | None = None, n_per_mode: int = 2) -> None:
    eval_mode = records[0].get("eval_mode", "unknown")
    print(f"\n{'=' * 68}\n  Flagged examples [{eval_mode}] (up to {n_per_mode} per failure mode)\n{'=' * 68}")
    for label in labels:
        flagged = [r for r in records if label in r["labels"]]
        if not flagged:
            continue
        print(f"\n  [{label}]  — {len(flagged)} flagged")
        for r in flagged[:n_per_mode]:
            meta = " | ".join(str(field_value(r, f, derived)) for f in group_fields if field_value(r, f, derived) is not None)
            text = r.get("criterion_text") or r["rubric_text"]
            print(f"    [{meta}]" if meta else "", end="\n" if meta else "")
            print(f"    {text[:140]}")
            just = next((v["justification"] for run in r.get("votes", []) for v in run if v["label"] == label), None)
            if just:
                print(f"    -> {just[:200]}")


def analyze_level(records: list[dict], model: str, level: str, group_fields: list[str], derived: dict | None = None, max_groups: int = 8) -> None:
    labels = records[0].get("included_failure_modes", [])
    print_table(f"Overall prevalence [{level}]  (n={len(records)}, judge: {model})", {level: records}, labels)
    for fname in group_fields:
        values = [field_value(r, fname, derived) for r in records]
        if all(v is None for v in values):
            continue
        order = [v for v, _ in Counter(v for v in values if v is not None).most_common(max_groups)]
        try:
            order = sorted(order)
        except TypeError:
            pass
        print_table(f"By {fname} [{level}]", {v: [r for r in records if field_value(r, fname, derived) == v] for v in order}, labels)
    print_flagged_examples(records, labels, group_fields, derived)


def analyze(records: list[dict], model: str, group_fields: list[str], derived_fields: dict | None = None) -> None:
    seen = [r.get("eval_mode") for r in records]
    order = [m for m in ("joined", "per_criterion") if m in seen] + sorted({m for m in seen if m not in ("joined", "per_criterion")})
    for mode in order:
        subset = [r for r in records if r.get("eval_mode") == mode]
        if subset:
            analyze_level(subset, model, mode, group_fields, derived_fields)
