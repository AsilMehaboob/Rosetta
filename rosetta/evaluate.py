"""Evaluation harness: ask every benchmark question several times and score the answers as data.

    python -m rosetta.evaluate --runs 2

Each benchmark question either has a reference query, whose rows are the expected answer, or is
marked as one the data cannot answer. Every run asks the model afresh (cache reads off) through
the same pipeline the API uses, then compares Rosetta's rows with the expected rows. Column
names, column order and row order are ignored; only the values count. Results are written to
eval/results/<run id>/: every run (question, query, rows, time, model, verdict), the expected
rows, and a summary of how many answers were correct, consistent and fast.
"""

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from rosetta.adapters import get_adapter
from rosetta.adapters.base import SourceAdapter
from rosetta.answer.pipeline import answer_question
from rosetta.answer.retrieve import LayerIndex, load_index
from rosetta.config import Settings
from rosetta.learn.profile import to_json_value
from rosetta.llm import GeminiLLM, QuotaExhausted
from rosetta.storage import Storage

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "eval" / "benchmark.yaml"
RESULTS = ROOT / "eval" / "results"

DECIMALS = 4  # numbers equal to this many decimals are the same answer
SAVED_ROWS = 200  # rows kept per run in the results files; comparison always uses every row
FAST_SECONDS = 60.0  # a run within this time counts as fast
# personal-data columns are masked in the saved results unless EVAL_MASK_PERSONAL_DATA=false
MASK_PERSONAL_DATA = os.getenv("EVAL_MASK_PERSONAL_DATA", "true").strip().lower() not in ("false", "0", "no")

# a run is correct when its verdict is one of these
CORRECT = {"match", "match_extra_columns", "correct_refusal"}


# ---------------------------------------------------------------------------
# comparing answers as data


_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ]00:00:00(?:\.0+)?)?$")


def normalise(value: Any) -> Any:
    """One comparable form per value: numbers rounded, dates as YYYY-MM-DD, text trimmed."""
    value = to_json_value(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
    text = str(value).strip()
    if m := _DAY.match(text):
        return m.group(1)
    if m := _MONTH.match(text):  # a month written as 2026-07 is the month starting 2026-07-01
        return f"{m.group(1)}-{m.group(2)}-01"
    return text


def _order(value: Any) -> tuple:
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, float(value), "")
    return (2, 0.0, str(value))


def _sorted_rows(rows: list[tuple]) -> list[tuple]:
    return sorted(rows, key=lambda r: [_order(v) for v in r])


def _project(rows: list[list], columns: list[int]) -> list[tuple]:
    return _sorted_rows([tuple(row[i] for i in columns) for row in rows])


def signature(rows: list[list]) -> str:
    """The same text for the same data, whatever the column order or row order."""
    norm = [[normalise(v) for v in row] for row in rows]
    width = len(norm[0]) if norm else 0
    order = sorted(range(width), key=lambda i: [_order(v) for v in sorted((r[i] for r in norm), key=_order)])
    return json.dumps(_project(norm, order), default=str)


def compare(expected_columns: list[str], expected_rows: list[list], columns: list[str], rows: list[list]) -> dict:
    """Do Rosetta's rows hold the expected data? Columns are paired by their values, never by their names."""
    want = [[normalise(v) for v in row] for row in expected_rows]
    got = [[normalise(v) for v in row] for row in rows]
    if len(got) != len(want):
        return {"verdict": "mismatch", "reason": f"{len(got)} rows, expected {len(want)}"}
    if not want:
        return {"verdict": "match", "reason": "both empty"}

    want_cols = [Counter(r[j] for r in want) for j in range(len(expected_columns))]
    got_cols = [Counter(r[k] for r in got) for k in range(len(columns))]
    candidates = [[k for k in range(len(columns)) if got_cols[k] == want_cols[j]] for j in range(len(expected_columns))]

    # pair as many expected columns as possible, each with a different column holding the same values,
    # such that the paired columns also line up row by row
    best: dict[int, int] = {}

    def search(j: int, pairs: dict[int, int]) -> None:
        nonlocal best
        if len(pairs) + (len(expected_columns) - j) <= len(best):
            return
        if j == len(expected_columns):
            best = dict(pairs)
            return
        for k in candidates[j]:
            if k in pairs.values():
                continue
            trial = {**pairs, j: k}
            if _project(want, list(trial)) == _project(got, list(trial.values())):
                search(j + 1, trial)
                if len(best) == len(expected_columns):
                    return
        search(j + 1, pairs)

    search(0, {})

    # a column with one value in every row only repeats the filter (such as the date asked about)
    constant = {j for j in range(len(expected_columns)) if len(want) > 1 and len(want_cols[j]) == 1}
    missing = [j for j in range(len(expected_columns)) if j not in best]
    missing_info = [expected_columns[j] for j in missing if j not in constant]
    extra = [columns[k] for k in range(len(columns)) if k not in best.values()]
    paired = {expected_columns[j]: columns[k] for j, k in best.items()}
    detail = {"paired_columns": paired, "extra_columns": extra}

    if not missing_info:
        notes = []
        if missing:
            notes.append("did not repeat the fixed column " + ", ".join(expected_columns[j] for j in missing))
        if extra:
            notes.append("added " + ", ".join(extra))
        verdict = "match_extra_columns" if extra else "match"
        return {"verdict": verdict, "reason": "; ".join(notes) or "same data", **detail}
    # Rosetta's unpaired figures are its own different numbers, not a column it chose not to show
    own_figures = [
        columns[k] for k in range(len(columns))
        if k not in best.values() and all(isinstance(r[k], float) or r[k] is None for r in got)
        and any(isinstance(r[k], float) for r in got)
    ]
    if best and not own_figures:
        return {"verdict": "partial", "reason": "right rows, but no column holds " + ", ".join(missing_info), **detail}

    reasons = []
    for j in missing:
        values = [r[j] for r in want]
        if all(isinstance(v, float) for v in values):
            reasons.append(f"{expected_columns[j]} totals {sum(values):,.0f} expected")
        else:
            reasons.append(f"no column holds the values of {expected_columns[j]}")
    totals = [
        f"{columns[k]} totals {sum(r[k] for r in got):,.0f}"
        for k in range(len(columns))
        if all(isinstance(r[k], float) for r in got)
    ]
    return {"verdict": "mismatch", "reason": "; ".join(reasons + totals), **detail}


# ---------------------------------------------------------------------------
# explaining a verdict: what exactly differs, in plain words

MAX_DIFF_ROWS = 50  # differing rows listed per run
MAX_EXAMPLES = 3  # differing rows quoted in the explanation


def _is_number(value: Any) -> bool:
    return isinstance(value, float)


def _shown(value: Any) -> str:
    if value is None:
        return "empty"
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,.4g}"
    return str(value)


def _number_columns(rows: list[list], width: int) -> set[int]:
    return {j for j in range(width) if rows and all(_is_number(r[j]) or r[j] is None for r in rows)
            and any(_is_number(r[j]) for r in rows)}


def _key_pairs(want: list[list], got: list[list], w_keys: list[int], g_keys: list[int], paired: dict[int, int]) -> dict[int, int]:
    """Each expected label column with the result column sharing most of its values."""
    pairs = {j: k for j, k in paired.items() if j in w_keys}
    for j in w_keys:
        if j in pairs:
            continue
        values = {r[j] for r in want}
        best, best_share = None, 0.5
        for k in g_keys:
            if k in pairs.values():
                continue
            share = len(values & {r[k] for r in got}) / max(len(values), 1)
            if share > best_share:
                best, best_share = k, share
        if best is not None:
            pairs[j] = best
    return pairs


def diagnose(
    expected_columns: list[str], expected_rows: list[list], columns: list[str], rows: list[list], verdict: dict
) -> dict:
    """A headline, a few plain facts, and the rows whose figures differ, for one run.

    Rows are lined up on their label columns (names, codes, dates); figures are then compared row by
    row, so the explanation can say which rows are missing or extra and which figures differ, by how
    much, and whether they differ in a pattern (always lower, always a whole multiple).
    """
    v = verdict["verdict"]
    if v == "correct_refusal":
        return {"headline": "Correctly said the data cannot answer this", "points": [verdict.get("reason", "")]}
    if v in ("wrong_refusal", "failed", "error", "answered_unanswerable"):
        headline = {
            "wrong_refusal": "Said the data cannot answer, but the reference query answers it",
            "failed": "No working query was found",
            "error": "The run stopped with an error",
            "answered_unanswerable": "Gave an answer although the data cannot support one",
        }[v]
        return {"headline": headline, "points": [p for p in [verdict.get("reason", "")] if p]}

    want = [[normalise(x) for x in r] for r in expected_rows]
    got = [[normalise(x) for x in r] for r in rows]
    e_index = {c: j for j, c in enumerate(expected_columns)}
    g_index = {c: k for k, c in enumerate(columns)}
    paired = {
        e_index[e]: g_index[g]
        for e, g in (verdict.get("paired_columns") or {}).items()
        if e in e_index and g in g_index
    }
    missing_cols = [expected_columns[j] for j in range(len(expected_columns)) if j not in paired]
    extra_cols = list(verdict.get("extra_columns") or [c for k, c in enumerate(columns) if k not in paired.values()])
    points: list[str] = []

    if v in ("match", "match_extra_columns"):
        headline = f"Same data as the reference ({len(got):,} row{'s' if len(got) != 1 else ''})"
        if extra_cols:
            points.append("Rosetta also showed " + ", ".join(extra_cols) + "; the figures do not change")
        if missing_cols:
            points.append("The reference repeats a fixed value in " + ", ".join(missing_cols) + "; Rosetta left it out")
        return {"headline": headline, "points": points}

    w_nums = _number_columns(want, len(expected_columns))
    g_nums = _number_columns(got, len(columns))
    w_keys = [j for j in range(len(expected_columns)) if j not in w_nums]
    g_keys = [k for k in range(len(columns)) if k not in g_nums]
    keys = _key_pairs(want, got, w_keys, g_keys, paired)
    key_label = " / ".join(expected_columns[j] for j in keys) or "row"

    def key_of(row: list, side: str) -> tuple:
        return tuple(row[j] if side == "w" else row[keys[j]] for j in keys)

    w_by = {key_of(r, "w"): r for r in want}
    g_by = {key_of(r, "g"): r for r in got}
    lined_up = len(w_by) == len(want) and len(g_by) == len(got) and keys

    if not lined_up:
        if len(got) != len(want):
            points.append(verdict.get("reason", ""))
            return {"headline": f"Rosetta returned {len(got):,} rows, the reference {len(want):,}",
                    "points": [p for p in points if p]}
        # same number of rows, but the labels are written differently: show both versions
        figures_ok = all(j in paired for j in w_nums)
        for j in w_keys:
            if j in keys:
                continue
            others = [k for k in g_keys if k not in keys.values()]
            sample = want[0][j]
            points.append(
                f"The reference labels rows by {expected_columns[j]} (for example “{_shown(sample)}”); "
                + (f"Rosetta by {', '.join(columns[k] for k in others)} (for example “{_shown(got[0][others[0]])}”)"
                   if others else "Rosetta has no such label")
            )
        if figures_ok and w_nums:
            points.append("The figures match row for row: " + ", ".join(
                f"{expected_columns[j]} ↔ {columns[paired[j]]}" for j in sorted(w_nums)))
        headline = (
            f"Same {len(got):,} rows and figures; the labels are written differently" if figures_ok
            else f"Same number of rows ({len(got):,}), but they could not be lined up with the reference"
        )
        return {"headline": headline, "points": [p for p in points if p]}

    missing = [k for k in w_by if k not in g_by]
    extra = [k for k in g_by if k not in w_by]
    both = [k for k in w_by if k in g_by]
    label = lambda k: " / ".join(_shown(x) for x in k)  # noqa: E731

    # figures: each expected figure with the result figure that agrees on most shared rows
    def cost(j: int, k: int) -> tuple:
        agree = sum(w_by[x][j] == g_by[x][k] for x in both)
        gap = sum(abs((g_by[x][k] or 0) - (w_by[x][j] or 0)) / (abs(w_by[x][j] or 0) + 1) for x in both)
        return (-agree, gap)

    measures = [(j, paired[j]) for j in sorted(w_nums) if j in paired]
    free_w = [j for j in sorted(w_nums) if j not in paired]
    free_g = [k for k in sorted(g_nums) if k not in paired.values()]
    for _, j, k in sorted((cost(j, k), j, k) for j in free_w for k in free_g):
        if j in {m[0] for m in measures} or k in {m[1] for m in measures}:
            continue
        measures.append((j, k))
    measures.sort()

    diff_rows = []
    for x in both:
        cells = []
        for j, k in measures:
            a, b = g_by[x][k], w_by[x][j]
            if a != b:
                cells.append({"column": expected_columns[j], "rosetta": a, "reference": b,
                              "difference": (a - b) if _is_number(a) and _is_number(b) else None})
        if cells:
            diff_rows.append({"key": label(x), "cells": cells})
    diff_rows.sort(key=lambda r: -max(abs(c["difference"] or 0) for c in r["cells"]))

    if missing or extra:
        headline = f"Rosetta returned {len(got):,} rows, the reference {len(want):,}"
        if missing:
            shown = ", ".join(label(k) for k in missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            points.append(f"Missing from Rosetta ({len(missing)}): {shown}")
        if extra:
            shown = ", ".join(label(k) for k in extra[:5]) + (f" and {len(extra) - 5} more" if len(extra) > 5 else "")
            points.append(f"Only in Rosetta ({len(extra)}): {shown}")
        if both:
            points.append(
                f"The {len(both):,} rows found on both sides agree" if not diff_rows
                else f"{len(diff_rows):,} of the {len(both):,} rows found on both sides have different figures"
            )
    elif diff_rows:
        changed = sorted({c["column"] for r in diff_rows for c in r["cells"]})
        verb = "differs" if len(changed) == 1 else "differ"
        headline = f"Same {len(both):,} {key_label} values; {', '.join(changed)} {verb} on {len(diff_rows):,} of them"
    elif v == "partial":
        headline = "Every row and every column Rosetta returned matches the reference"
        if missing_cols:
            points.append("The reference also returns " + ", ".join(missing_cols) + ", which Rosetta did not show")
    else:
        headline = "Same rows; the figures could not be paired with the reference"
        points.append(verdict.get("reason", ""))

    for r in diff_rows[:MAX_EXAMPLES]:
        c = r["cells"][0]
        points.append(f"{r['key']}: Rosetta {_shown(c['rosetta'])}, reference {_shown(c['reference'])}")
    for j, k in measures:
        total_g = sum(r[k] for r in got if _is_number(r[k]))
        total_w = sum(r[j] for r in want if _is_number(r[j]))
        has_gaps = any(r[j] is None for r in want) or any(r[k] is None for r in got)
        if not has_gaps and total_g != total_w and any(c["column"] == expected_columns[j] for r in diff_rows for c in r["cells"]):
            points.append(
                f"Total {expected_columns[j]}: Rosetta {_shown(total_g)}, reference {_shown(total_w)} "
                f"({'+' if total_g > total_w else '−'}{_shown(abs(total_g - total_w))})"
            )
    cells = [c for r in diff_rows for c in r["cells"] if c["difference"] is not None]
    if len(cells) >= 2:
        if all(c["rosetta"] and c["reference"] / c["rosetta"] >= 2 and (c["reference"] / c["rosetta"]).is_integer() for c in cells):
            points.append("On every differing row the reference is a whole multiple (2×, 3×…) of Rosetta's figure, "
                          "a sign that the reference counts some rows more than once")
        elif all(c["difference"] < 0 for c in cells):
            points.append("Rosetta's figure is lower on every differing row: it left out something the reference counts")
        elif all(c["difference"] > 0 for c in cells):
            points.append("Rosetta's figure is higher on every differing row: it counts something the reference leaves out")
    used = set(keys) | {j for j, _ in measures}
    unused = [c for c in missing_cols if e_index[c] not in used]
    if unused and v != "partial":
        points.append("No Rosetta column holds the reference's " + ", ".join(unused))
    return {"headline": headline, "points": [p for p in points if p], "differences": diff_rows[:MAX_DIFF_ROWS]}


# ---------------------------------------------------------------------------
# running the benchmark


def load_benchmark(path: Path = BENCHMARK) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pinned(sql: str, now_token: str | None, adapter: SourceAdapter, reference: date) -> str:
    """The reference query with its "now" replaced by the reference date, at the current time of day."""
    if not now_token:
        return sql
    moment = datetime.combine(reference, datetime.now().time().replace(microsecond=0))
    return re.sub(rf"\b{re.escape(now_token)}\b", adapter.datetime_literal(moment), sql, flags=re.IGNORECASE)


def _sensitive_columns(index: LayerIndex) -> set[str]:
    return {c["name"].upper() for t in index.tables.values() for c in t["columns"] if c.get("sensitive")}


def _for_saving(columns: list[str], rows: list[list], sensitive: set[str]) -> list[list]:
    """Rows as written to the results files: capped, with personal-data columns masked (unless turned off)."""
    hide = {i for i, c in enumerate(columns) if c.upper() in sensitive} if MASK_PERSONAL_DATA else set()
    return [["***" if i in hide and v is not None else v for i, v in enumerate(row)] for row in rows[:SAVED_ROWS]]


def score(question: dict, result: dict, expected: dict | None) -> dict:
    """The verdict for one run."""
    status = result.get("status")
    if question.get("expect") == "cannot_answer":
        if status == "cannot_answer":
            return {"verdict": "correct_refusal", "reason": result.get("answer", "")}
        if status == "answered":
            return {"verdict": "answered_unanswerable", "reason": "gave an answer the data cannot support"}
        return {"verdict": "failed", "reason": result.get("answer", "")}
    if status == "error":
        return {"verdict": "error", "reason": result.get("error", "")}
    if status == "cannot_answer":
        return {"verdict": "wrong_refusal", "reason": result.get("answer", "")}
    if status != "answered":
        return {"verdict": "failed", "reason": result.get("answer", "")}
    if expected is None:
        return {"verdict": "error", "reason": "the benchmark gives no reference query for this question"}
    return compare(expected["columns"], expected["rows"], result["columns"], result["rows"])


def run_benchmark(
    runs: int = 2,
    only: list[str] | None = None,
    reference: date | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Ask every question `runs` times; write and return the summary."""
    settings = Settings.from_env()
    storage = Storage(settings.data_dir)
    reference = reference or settings.reference_date or date.today()
    benchmark = load_benchmark()
    questions = [q for q in benchmark["questions"] if not only or q["id"] in only]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)

    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, storage, use_cache=False)
    adapter = get_adapter(settings)
    index = load_index(storage)
    sensitive = _sensitive_columns(index)
    started = datetime.now()
    total = runs * len(questions)
    state = {
        "run_id": run_id,
        "done": 0,
        "total": total,
        "question": None,
        "run": None,
        "last": None,
        "questions": [q["id"] for q in questions],
        "runs_per_question": runs,
        "completed": [],  # {question, run, verdict} of every run finished so far
    }
    records: list[dict] = []
    stopped = None

    def report(**changes: Any) -> None:
        state.update(changes)
        if progress:
            progress({**state, "completed": list(state["completed"])})

    try:
        expected: dict[str, dict] = {}
        for q in questions:
            if q.get("reference_sql"):
                sql = _pinned(q["reference_sql"], benchmark.get("now_token"), adapter, reference)
                result = adapter.run_select(sql, max_rows=100_000)
                expected[q["id"]] = {
                    "sql": sql,
                    "columns": list(result.columns),
                    "rows": [[to_json_value(v) for v in row] for row in result.rows],
                }
        with open(out / "expected.json", "w", encoding="utf-8") as f:
            saved = {
                k: {**e, "row_count": len(e["rows"]), "rows": _for_saving(e["columns"], e["rows"], sensitive)}
                for k, e in expected.items()
            }
            json.dump(saved, f, ensure_ascii=False, indent=2)

        with open(out / "runs.jsonl", "w", encoding="utf-8") as log:
            for run in range(1, runs + 1):
                for q in questions:
                    report(question=q["id"], run=run)
                    t0 = time.perf_counter()
                    try:
                        result = answer_question(llm, adapter, storage, index, q["question"], reference)
                    except QuotaExhausted:
                        raise
                    except Exception as exc:  # one broken run is a result, not the end of the evaluation
                        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                  "seconds": round(time.perf_counter() - t0, 2)}
                    verdict = score(q, result, expected.get(q["id"]))
                    columns, rows = result.get("columns") or [], result.get("rows") or []
                    e = expected.get(q["id"]) or {"columns": [], "rows": []}
                    verdict["diagnosis"] = diagnose(e["columns"], e["rows"], columns, rows, verdict)
                    record = {
                        "question_id": q["id"],
                        "run": run,
                        "question": q["question"],
                        "status": result.get("status"),
                        **verdict,
                        "correct": verdict["verdict"] in CORRECT,
                        "sql": result.get("sql"),
                        "columns": columns,
                        "rows": _for_saving(columns, rows, sensitive),
                        "row_count": len(rows),
                        "signature": signature(rows) if result.get("status") == "answered" else result.get("status"),
                        "answer": result.get("answer"),
                        "explanation": result.get("explanation"),
                        "seconds": result.get("seconds"),
                        "llm_calls": result.get("llm_calls"),
                        "model": llm.model,
                        "layer_version": index.version,
                        "reference_date": reference.isoformat(),
                    }
                    records.append(record)
                    log.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    log.flush()
                    state["completed"].append({"question": q["id"], "run": run, "verdict": verdict["verdict"]})
                    report(done=state["done"] + 1, question=None, last=f"{q['id']} run {run}: {verdict['verdict']}")
    except QuotaExhausted as exc:
        stopped = str(exc)
    finally:
        adapter.close()

    summary = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runs_per_question": runs,
        "reference_date": reference.isoformat(),
        "model": llm.model,
        "layer_version": index.version,
        "fast_seconds": FAST_SECONDS,
        "stopped": stopped,
        **metrics(questions, records, runs),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    report(question=None, run=None, last=None, finished=True)
    return summary


# ---------------------------------------------------------------------------
# metrics


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _seconds(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))], 2),
        "max": round(ordered[-1], 2),
    }


def metrics(questions: list[dict], records: list[dict], runs: int) -> dict:
    """How many answers were correct, consistent and fast, per question and overall."""
    per_question = []
    for q in questions:
        mine = [r for r in records if r["question_id"] == q["id"]]
        times = [r["seconds"] for r in mine if r.get("seconds") is not None]
        per_question.append(
            {
                "id": q["id"],
                "level": q.get("level"),
                "question": q["question"],
                "checks": q.get("checks"),
                "note": q.get("note"),
                "expect": q.get("expect", "answer"),
                "runs": len(mine),
                "correct_runs": sum(r["correct"] for r in mine),
                "verdicts": [r["verdict"] for r in mine],
                "accuracy": _rate(sum(r["correct"] for r in mine), len(mine)),
                "consistent": len(mine) == runs and len({r["signature"] for r in mine}) == 1,
                "seconds": _seconds(times),
                "llm_calls_mean": round(statistics.fmean([r["llm_calls"] or 0 for r in mine]), 1) if mine else None,
            }
        )

    unanswerable = {q["id"] for q in questions if q.get("expect") == "cannot_answer"}
    answerable = [r for r in records if r["question_id"] not in unanswerable]
    refusal = [r for r in records if r["question_id"] in unanswerable]
    times = [r["seconds"] for r in records if r.get("seconds") is not None]
    wrong_numbers = [r for r in records if r["verdict"] in ("mismatch", "answered_unanswerable")]
    return {
        "overall": {
            "questions": len(questions),
            "runs": len(records),
            "accuracy": _rate(sum(r["correct"] for r in records), len(records)),
            "answer_accuracy": _rate(sum(r["correct"] for r in answerable), len(answerable)),
            "solved_every_run": _rate(sum(p["correct_runs"] == runs for p in per_question), len(per_question)),
            "consistency": _rate(sum(p["consistent"] for p in per_question), len(per_question)),
            "query_success": _rate(sum(r["status"] == "answered" for r in answerable), len(answerable)),
            "partial_runs": sum(r["verdict"] == "partial" for r in records),
            "refusal_accuracy": _rate(sum(r["verdict"] == "correct_refusal" for r in refusal), len(refusal)),
            "false_refusals": _rate(sum(r["verdict"] == "wrong_refusal" for r in answerable), len(answerable)),
            "wrong_answers": _rate(len(wrong_numbers), len(records)),
            "fast": _rate(sum(t <= FAST_SECONDS for t in times), len(times)),
            "seconds": _seconds(times),
            "llm_calls": sum(r["llm_calls"] or 0 for r in records),
        },
        "per_question": per_question,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def to_markdown(summary: dict) -> str:
    o = summary["overall"]
    lines = [
        f"# Evaluation {summary['run_id']}",
        "",
        f"Model `{summary['model']}` · layer `{summary['layer_version']}` · today = {summary['reference_date']} · "
        f"{summary['runs_per_question']} runs per question · {summary['started_at']} to {summary['finished_at']}",
        "",
    ]
    if summary.get("stopped"):
        lines += [f"**Stopped early:** {summary['stopped']}", ""]
    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Correct runs (all questions) | {_pct(o['accuracy'])} of {o['runs']} |",
        f"| Correct runs (answerable questions) | {_pct(o['answer_accuracy'])} |",
        f"| Questions correct in every run | {_pct(o['solved_every_run'])} |",
        f"| Questions with the same result in every run | {_pct(o['consistency'])} |",
        f"| Queries that ran (answerable questions) | {_pct(o['query_success'])} |",
        f"| Right rows but a column missing | {o['partial_runs']} runs |",
        f"| Unanswerable questions refused | {_pct(o['refusal_accuracy'])} |",
        f"| Answerable questions wrongly refused | {_pct(o['false_refusals'])} |",
        f"| Runs that gave a wrong answer | {_pct(o['wrong_answers'])} |",
        f"| Runs within {summary['fast_seconds']:.0f}s | {_pct(o['fast'])} |",
        f"| Seconds per run (mean / median / p95 / max) | {o['seconds']['mean']} / {o['seconds']['median']} / "
        f"{o['seconds']['p95']} / {o['seconds']['max']} |",
        f"| Model calls | {o['llm_calls']} |",
        "",
        "| # | Level | Question | Verdicts per run | Correct | Consistent | Median s |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in summary["per_question"]:
        lines.append(
            f"| {p['id']} | {p['level']} | {p['question']} | {', '.join(p['verdicts'])} | "
            f"{p['correct_runs']}/{p['runs']} | {'yes' if p['consistent'] else 'no'} | {p['seconds']['median']} |"
        )
    notes = [p for p in summary["per_question"] if p.get("note")]
    if notes:
        lines += ["", "## Notes", ""] + [f"- **{p['id']}**: {p['note'].strip()}" for p in notes]
    lines += ["", "Every run (query, rows, time, verdict and reason) is in `runs.jsonl`; expected rows in `expected.json`."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# reading saved results (for the API and the UI)


def list_results() -> list[dict]:
    """Saved evaluations, newest first: their summaries."""
    found = []
    for path in sorted(RESULTS.glob("*/summary.json"), reverse=True):
        with open(path, encoding="utf-8") as f:
            found.append(json.load(f))
    return found


def read_result(run_id: str) -> dict | None:
    """One saved evaluation: its summary, every run, and the expected rows."""
    if not re.fullmatch(r"\d{8}-\d{6}", run_id):
        return None
    folder = RESULTS / run_id
    if not (folder / "summary.json").exists():
        return None
    with open(folder / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(folder / "runs.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    expected_path = folder / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else {}
    for r in records:  # evaluations saved before diagnoses existed get one from their saved rows
        if "diagnosis" not in r:
            e = expected.get(r["question_id"]) or {"columns": [], "rows": []}
            try:
                r["diagnosis"] = diagnose(e["columns"], e["rows"], r.get("columns") or [], r.get("rows") or [], r)
            except Exception:
                r["diagnosis"] = {"headline": r.get("reason", ""), "points": []}
    return {"summary": summary, "runs": records, "expected": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark questions and score the answers as data.")
    parser.add_argument("--runs", type=int, default=2, help="times each question is asked (default 2)")
    parser.add_argument("--only", nargs="*", help="question ids to run, such as q1 q3 (default: all)")
    parser.add_argument("--reference-date", type=date.fromisoformat, help='"today" for the run (default: REFERENCE_DATE or today)')
    args = parser.parse_args()

    def show(state: dict) -> None:
        if state["question"]:
            print(f"[{state['done'] + 1}/{state['total']}] run {state['run']} · {state['question']} …", flush=True)
        elif state["last"]:
            print(f"  {state['last']}", flush=True)

    summary = run_benchmark(args.runs, args.only, args.reference_date, show)
    print(to_markdown(summary))
    print(f"Saved to {RESULTS / summary['run_id']}")


if __name__ == "__main__":
    main()
