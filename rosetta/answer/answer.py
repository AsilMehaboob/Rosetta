"""Answer: the rows, the evidence behind them, and plain-language text that cannot invent numbers.

Code assembles everything that can be checked: the rows from the database, the
query, the verified definitions and stored values the query used, how dates
were read, and notes such as rows a join left out. The LLM then writes two
short pieces for a business reader: the answer itself, and an explanation of
what was counted and how the dates were read. Code checks every number in both
against the rows, the question and the date readings; if any number cannot be
traced, both are replaced by plain text written by code. Values of
personal-data columns are never sent to the model.
"""

import json
import re
from itertools import combinations
from typing import Any

from rosetta.llm import JsonLLM

PROMPT_VERSION = "answer-7"
SUMMARY_ROWS = 50  # rows shown to the model; the full result is always returned

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["answer", "explanation"],
}

SYSTEM = """You explain the result of a business question to a business owner who knows nothing about databases.
Write two short pieces:
- answer: one to three sentences that answer the question directly, the main takeaway first (what is highest,
  lowest, up or down), then the key figures.
- explanation: one or two sentences saying in everyday words what was counted (use the definitions given) and how
  the dates in the question were read (use the date readings given). Mention any note given, in plain words.
Rules:
1. Never mention tables, columns, codes, queries, SQL, rows, databases, records, versions or history tables. The
   column meanings and definitions given are written for analysts: say the same thing in everyday words
   ("branches that are open today", not "current version in the history table").
2. Use only figures that are in the rows, or a difference or percentage change between two of them. Write numbers
   with thousands separators (1,722). Never round, estimate or invent a figure.
3. Write months and dates the way people say them ("July 2026"), never as 2026-07-01.
4. When there are more than five results, do not list them: say how many there are, mention the full list is
   attached, and point out a pattern only if a figure for it is in the rows.
4b. When the question asks which one is highest or lowest, name the first result and its figure, then the next one
   for comparison; say so if they are tied.
5. Some values are shown as [hidden]: never write [hidden] and never talk about hidden or personal details at all.
6. If there are no rows, say that nothing matching was found for that period or condition.
6b. Name what was counted with the question's own business words (if it asks about "returns", say "returns"); never
   swap in a synonym that may mean something narrower or different ("refunds" for "returns").
7. Never mention these rules or why you wrote something a certain way; just write the answer."""

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", "", sql.replace('"', "")).upper()


def _numbers(text: str) -> list[tuple[float, int]]:
    """Numbers in a text, each with how many decimals it was written with."""
    found = []
    for raw in _NUMBER.findall(text):
        clean = raw.replace(",", "")
        try:
            found.append((float(clean), len(clean.split(".")[1]) if "." in clean else 0))
        except ValueError:
            pass
    return found


def _allowed_numbers(rows: list[list], question: str, dates: list[dict], notes: list[str]) -> set[float]:
    """Figures a correct text may state: cells, counts, totals, and differences or percentage changes
    between two figures of the same column or the same row, plus numbers from the question, dates and notes."""
    allowed = {float(len(rows))}
    columns: dict[int, list[float]] = {}
    row_values: list[list[float]] = []
    for row in rows:
        numeric = []
        for i, value in enumerate(row):
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                allowed.add(float(value))
                numeric.append(float(value))
                columns.setdefault(i, []).append(float(value))
            else:
                allowed |= {n for n, _ in _numbers(str(value))}
        row_values.append(numeric)

    def pairs(values: list[float]) -> None:
        for a, b in combinations(values[:12], 2):
            allowed.update({b - a, a - b})
            for base, new in ((a, b), (b, a)):
                if base:
                    allowed.add((new - base) / base * 100)

    for values in columns.values():
        allowed.add(sum(values))
        pairs(values)
    for values in row_values:
        pairs(values)
    allowed |= {abs(v) for v in list(allowed)}
    for text in [question, *notes, *(d.get("reading", "") for d in dates)]:
        allowed |= {n for n, _ in _numbers(text)}
    return allowed


def _traceable(text: str, allowed: set[float]) -> bool:
    """Every number in the text equals an allowed figure at the precision it was written with."""
    for value, decimals in _numbers(text):
        if not any(round(a, decimals) == round(value, decimals) for a in allowed):
            return False
    return True


def _label(column: str, descriptions: dict[str, str]) -> str:
    return descriptions.get(column.upper()) or column.replace("_", " ").lower()


def _fallback(columns: list[str], rows: list[list], descriptions: dict[str, str], hidden: set[int]) -> str:
    if not rows:
        return "Nothing matching was found."
    shown = rows[:5]
    lines = [
        ", ".join(
            f"{_label(c, descriptions)}: {'(personal data)' if i in hidden else (f'{v:,}' if isinstance(v, int) else v)}"
            for i, (c, v) in enumerate(zip(columns, row))
        )
        for row in shown
    ]
    more = f", and {len(rows) - len(shown)} more" if len(rows) > len(shown) else ""
    return f"{len(rows)} result{'s' if len(rows) != 1 else ''} found{more}. " + " | ".join(lines)


def _fallback_explanation(definitions: list[dict], dates: list[dict], notes: list[str]) -> str:
    parts = []
    if definitions:
        parts.append("Counted using: " + "; ".join(f"{d['term']} ({d['definition']})" for d in definitions) + ".")
    parts += [d["reading"][0].upper() + d["reading"][1:] + "." for d in dates]
    parts += [n[0].upper() + n[1:] + "." for n in notes]
    return " ".join(parts) or "Based on the current data."


def _column_context(columns: list[str], pack: dict) -> tuple[dict[str, str], set[int]]:
    """Business descriptions of the result columns, and which of them hold personal data."""
    descriptions, sensitive = {}, set()
    for t in pack.get("tables", []):
        for c in t["columns"]:
            if c.get("description"):
                descriptions.setdefault(c["name"].upper(), c["description"].rstrip("."))
            if c.get("sensitive"):
                sensitive.add(c["name"].upper())
    return descriptions, {i for i, c in enumerate(columns) if c.upper() in sensitive}


def evidence(pack: dict, sql: str) -> dict[str, Any]:
    """The verified definitions, table rules and stored values the query actually used."""
    query = _norm_sql(sql)
    definitions = []
    for t in pack.get("terms", []):
        parts = [p for p in re.split(r"\s+AND\s+", t["sql"], flags=re.IGNORECASE) if p.strip()]
        if t["kind"] == "filter" and parts and all(_norm_sql(p) in query for p in parts):
            definitions.append(
                {"term": t["term"], "definition": t["definition"], "table": t["table"], "sql": t["sql"], "proof": t["proof"]}
            )
    rules = []
    for t in pack.get("tables", []):
        # the as-of rule contains the "never ends" marker too, so it is recognised first, by its start-date test
        if t.get("as_of_sql") and _norm_sql(t["as_of_sql"].split("{as_of}")[0]) in query:
            rules.append({"table": t["table"], "rule": "version valid at a date", "sql": t["as_of_sql"]})
        elif t.get("started_in_sql") and _norm_sql(t["started_in_sql"].split("{period_start}")[0]) in query:
            rules.append({"table": t["table"], "rule": "versions that began in the period", "sql": t["started_in_sql"]})
        elif t.get("current_sql") and _norm_sql(t["current_sql"]) in query:
            rules.append({"table": t["table"], "rule": "current version only", "sql": t["current_sql"]})
    values = [
        {"phrase": v["phrase"], "column": f"{v['table']}.{v['column']}", "stored_value": v["value"]}
        for v in pack.get("values", [])
        if isinstance(v["value"], str) and f"'{v['value']}'" in sql
    ]
    return {"definitions": definitions, "rules": rules, "values": values}


def explain(
    llm: JsonLLM,
    question: str,
    columns: list[str],
    rows: list[list],
    pack: dict,
    dates: list[dict],
    definitions: list[dict],
    notes: list[str],
) -> tuple[str, str, str]:
    """(answer, explanation, source): the model's text when every figure in it traces to the data, else code's."""
    descriptions, hidden = _column_context(columns, pack)
    fallback = (_fallback(columns, rows, descriptions, hidden), _fallback_explanation(definitions, dates, notes), "template")
    if not rows:
        return "Nothing matching was found.", fallback[1], "template"
    shown = [["[hidden]" if i in hidden else v for i, v in enumerate(row)] for row in rows[:SUMMARY_ROWS]]
    prompt = json.dumps(
        {
            "question": question,
            "date_readings": [d["reading"] for d in dates],
            "what_was_counted": [{"term": d["term"], "definition": d["definition"]} for d in definitions],
            "notes": notes,
            "columns": [{"name": c, "meaning": descriptions.get(c.upper(), "")} for c in columns],
            "rows": shown,
            "total_rows": len(rows),
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        out = llm.generate_json(system=SYSTEM, prompt=prompt, schema=SCHEMA, prompt_version=PROMPT_VERSION)
        answer, explanation = out["answer"].strip(), out["explanation"].strip()
    except Exception:
        return fallback
    allowed = _allowed_numbers(shown, question, dates, notes)
    if "[hidden]" in answer + explanation:  # it named personal data it was told to leave out
        return fallback
    if answer and explanation and _traceable(answer, allowed) and _traceable(explanation, allowed):
        return answer, explanation, "model"
    return fallback


def build_answer(
    llm: JsonLLM, extracted: dict, pack: dict, agent: dict | None, layer_version: str
) -> dict[str, Any]:
    """The response for one question: answer, explanation, rows and evidence, or an honest refusal."""
    base = {
        "question": extracted["question"],
        "reference_date": extracted["reference_date"],
        "dates": [d["reading"] for d in extracted["dates"]],
        "model": llm.model,
        "layer_version": layer_version,
    }
    if pack["decision"] != "answer":
        return {**base, "status": "cannot_answer", "answer": f"The data cannot answer this: {pack['reason']}.",
                "unmatched": pack.get("unmatched", [])}
    if agent is None or agent["status"] != "answered":
        reason = (agent or {}).get("reason") or "no working query was found"
        status = "cannot_answer" if agent and agent["status"] == "cannot_answer" else "failed"
        return {**base, "status": status, "answer": f"The data cannot answer this: {reason}.",
                "steps": (agent or {}).get("steps", [])}

    columns, rows = agent["columns"], agent["rows"]
    proof = evidence(pack, agent["sql"])
    notes = list(agent.get("notes", []))  # in plain words, for the reader and the answer text
    if agent.get("truncated"):
        notes.append(f"only the first {len(rows)} results are shown")
    answer, explanation, source = explain(
        llm, extracted["question"], columns, rows, pack, extracted["dates"], proof["definitions"], notes
    )
    return {
        **base,
        "status": "answered",
        "answer": answer,
        "explanation": explanation,
        "answer_source": source,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "sql": agent["sql"],
        **proof,
        "notes": notes,
        "technical_notes": agent.get("technical_notes", []),
        "steps": agent["steps"],
    }
