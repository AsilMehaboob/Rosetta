"""SQL agent: write the query from the retrieved slice, with three tools.

The LLM sees only what Retrieve packed (tables, verified terms, exact values,
joins, dates already read), never the whole schema. Each step it picks one
action: run_sql (checked by sqlcheck, then run read-only with a timeout and a
row cap), get_column_values, search_knowledge, or cannot_answer. Errors and
empty results go back to it so it can fix the query. The rows of the answer
always come from the database, never from the model's text.
"""

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from rosetta.adapters.base import SourceAdapter
from rosetta.answer.retrieve import LayerIndex, tokens
from rosetta.answer.sqlcheck import check_query
from rosetta.learn.profile import to_json_value
from rosetta.llm import JsonLLM

PROMPT_VERSION = "sql-agent-1"


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 6
    max_rows: int = 1000
    values_max: int = 50  # values returned by get_column_values
    search_hits: int = 6


STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "action": {"type": "string", "enum": ["run_sql", "get_column_values", "search_knowledge", "cannot_answer"]},
        "sql": {"type": "string"},
        "table": {"type": "string"},
        "column": {"type": "string"},
        "query": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["thinking", "action"],
}

SYSTEM = """You answer one business question by writing a single {dialect} SQL query.
You get a context: the question, its dates already read into exact ranges, the tables that hold the answer,
verified terms (business definitions proven on the data), exact stored values, and the joins allowed.
Each step, reply with one action:
- run_sql (sql): run your query. It is checked and run read-only; you see errors or an empty result.
  A query that runs and returns rows is the final answer, so only run the query that answers the question.
- get_column_values (table, column): see the real values of a column.
- search_knowledge (query): look up terms, tables or columns not in the context; a full table name returns that table.
- cannot_answer (reason): when the data cannot answer the question. A wrong number is worse than no number.

Rules:
1. Use only tables and columns from the context or from your searches. Never invent names or values.
2. When a verified term matches words of the question, use its SQL as written.
3. Copy stored values exactly as given in "values" or seen with get_column_values (case, accents, spaces).
4. History tables keep versions. For the present, add the table's current_sql. For a past date, use its as_of_sql
   with {{as_of}} replaced by that date. For the state at the time of an event, join the history table on the entity
   and put the event's date column in place of {{as_of}}.
5. Snapshot tables repeat every entity once per snapshot date. For the present use the latest date; for a past
   date use that date. Never add up or count across several snapshot dates.
6. When asked how many of something, count distinct entities (COUNT(DISTINCT key)), not rows, unless each row is
   one of the things counted.
7. A date period from "dates" is start <= column < end, with the date literals exactly as given.
8. Join only along the joins given; when joining into a history or snapshot table, keep one row per key
   (its current, as-of or snapshot rule), or rows multiply.
9. Select only what the question asks for, with readable aliases; order the rows sensibly.
{dialect_rules}"""


def _date_sql(adapter: SourceAdapter, iso: str) -> str:
    return adapter.date_literal(date.fromisoformat(iso))


def build_context(extracted: dict, pack: dict, adapter: SourceAdapter) -> dict[str, Any]:
    dates = []
    for d in extracted["dates"]:
        entry = {"text": d["text"], "reading": d["reading"], "kind": d["kind"]}
        if d["kind"] == "point":
            entry["sql_at"] = _date_sql(adapter, d["at"])
        elif d["kind"] == "period":
            entry["periods"] = [
                {"start": _date_sql(adapter, p["start"]), "end": _date_sql(adapter, p["end"])} for p in d["periods"]
            ]
        dates.append(entry)
    return {
        "question": extracted["question"],
        "reference_date": _date_sql(adapter, extracted["reference_date"]),
        "answer_shape": extracted["output"],
        "dates": dates,
        "tables": pack["tables"],
        "terms": pack["terms"],
        "values": [{k: v[k] for k in ("phrase", "table", "column", "value")} for v in pack["values"]],
        "joins": pack["joins"],
        "related_tables": [{"table": t, "kind": k, "description": d} for t, k, d in pack.get("related_tables", [])],
        "not_found_in_data": pack.get("unmatched", []),
    }


def table_brief(index: LayerIndex, key: str) -> dict[str, Any]:
    t = index.tables[key]
    columns = []
    for c in t["columns"]:
        entry = {k: c[k] for k in ("name", "type", "description") if c.get(k)}
        for k in ("values", "meanings", "sensitive"):
            if c.get(k):
                entry[k] = c[k]
        columns.append(entry)
    return {
        "table": key,
        "description": t["description"],
        "kind": t["kind"],
        "grain": (t["grain"] or {}).get("text"),
        "current_sql": t["current_sql"],
        "as_of_sql": t["as_of_sql"],
        "columns": columns,
        "terms": [
            {"term": x["term"], "kind": x["kind"], "sql": x["sql"], "proof": x["verification"]["proof"]}
            for x in index.terms.values()
            if x["table"] == key
        ],
    }


def search_knowledge(index: LayerIndex, query: str, hits: int) -> dict[str, Any]:
    key = next((k for k in index.tables if k.upper() == query.strip().upper() or k.split(".")[1] == query.strip().upper()), None)
    if key:
        return table_brief(index, key)
    words = tokens(query)
    top = lambda scores: sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))[:hits]  # noqa: E731
    return {
        "terms": [
            {"table": index.terms[i]["table"], "term": index.terms[i]["term"], "kind": index.terms[i]["kind"],
             "sql": index.terms[i]["sql"], "proof": index.terms[i]["verification"]["proof"]}
            for i, _ in top(index.term_index.score(words))
        ],
        "tables": [
            {"table": k, "kind": index.tables[k]["kind"], "description": index.tables[k]["description"]}
            for k, _ in top(index.table_index.score(words))
        ],
        "columns": [
            {"table": t, "column": c,
             "description": next(x["description"] for x in index.tables[t]["columns"] if x["name"] == c)}
            for (t, c), _ in top(index.column_index.score(words))
        ],
    }


def column_values(adapter: SourceAdapter, index: LayerIndex, table: str, column: str, limit: int) -> dict[str, Any]:
    key = next((k for k in index.tables if k.upper() == table.strip().upper() or k.split(".")[1] == table.strip().upper()), None)
    if key is None:
        return {"error": f"no table {table}"}
    col = next((c for c in index.tables[key]["columns"] if c["name"].upper() == column.strip().upper()), None)
    if col is None:
        return {"error": f"no column {column} in {key}"}
    if col.get("sensitive"):
        return {"error": f"{key}.{col['name']} holds personal data; its values are not shown"}
    schema, name = key.split(".", 1)
    q = adapter.quote
    sql = (
        f"SELECT {q(col['name'])}, COUNT(*) FROM {q(schema)}.{q(name)} WHERE {q(col['name'])} IS NOT NULL "
        f"GROUP BY {q(col['name'])} ORDER BY COUNT(*) DESC, {q(col['name'])} {adapter.limit_clause(limit)}"
    )
    rows = adapter.run_select(sql, max_rows=limit).rows
    return {"table": key, "column": col["name"], "values": [[to_json_value(v), int(n)] for v, n in rows]}


def run_agent(
    llm: JsonLLM,
    adapter: SourceAdapter,
    index: LayerIndex,
    profiles: dict[str, dict],
    extracted: dict,
    pack: dict,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or AgentConfig()
    context = build_context(extracted, pack, adapter)
    system = SYSTEM.format(dialect=adapter.sqlglot_dialect, dialect_rules=adapter.dialect_rules())
    allowed = {k: profiles[k] for k in index.tables if k in profiles}
    steps: list[dict] = []
    empty: dict | None = None  # an empty result gets one chance to be fixed

    for _ in range(cfg.max_steps):
        prompt = json.dumps({"context": context, "steps_so_far": steps}, sort_keys=True, ensure_ascii=False, default=str)
        out = llm.generate_json(system=system, prompt=prompt, schema=STEP_SCHEMA, prompt_version=PROMPT_VERSION)
        action = out.get("action")
        step = {k: v for k, v in out.items() if v not in (None, "")}

        if action == "cannot_answer":
            return {"status": "cannot_answer", "reason": out.get("reason", ""), "steps": steps + [step]}
        if action == "get_column_values":
            observation = column_values(adapter, index, out.get("table", ""), out.get("column", ""), cfg.values_max)
        elif action == "search_knowledge":
            observation = search_knowledge(index, out.get("query", ""), cfg.search_hits)
        elif action == "run_sql":
            sql = out.get("sql", "").strip().rstrip(";").strip()
            problems = check_query(sql, allowed, adapter.sqlglot_dialect)
            if problems:
                observation = {"error": "the query was not run", "problems": problems}
            else:
                try:
                    result = adapter.run_select(sql, max_rows=cfg.max_rows)
                except Exception as exc:
                    observation = {"error": adapter.map_error(exc)}
                else:
                    answer = {
                        "status": "answered",
                        "sql": sql,
                        "columns": list(result.columns),
                        "rows": [[to_json_value(v) for v in row] for row in result.rows],
                        "truncated": result.truncated,
                    }
                    if result.rows or empty is not None:
                        return {**answer, "steps": steps + [step]}
                    empty = answer
                    observation = {
                        "result": "0 rows",
                        "hint": "check the filters: exact stored values, date ranges, current/as-of rules and joins. "
                        "If 0 is the true answer, run the same query again.",
                    }
        else:
            observation = {"error": f"unknown action {action!r}"}
        steps.append({**step, "observation": observation})

    if empty is not None:
        return {**empty, "steps": steps}
    return {"status": "failed", "reason": f"no working query within {cfg.max_steps} steps", "steps": steps}
