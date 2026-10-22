"""SQL agent: write the query from the retrieved slice, with three tools.

The LLM sees only what Retrieve packed (tables, verified terms, exact values,
joins, dates already read), never the whole schema. Each step it picks one
action: run_sql (checked by sqlcheck, then run read-only with a timeout and a
row cap), get_column_values, search_knowledge, or cannot_answer. Errors and
empty results go back to it so it can fix the query. The rows of the answer
always come from the database, never from the model's text.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from rosetta.adapters.base import SourceAdapter
from rosetta.answer.retrieve import LayerIndex, _match_strength, _stem, fold, tokens
from rosetta.answer.sqlcheck import check_query
from rosetta.learn.profile import to_json_value
from rosetta.llm import JsonLLM

PROMPT_VERSION = "sql-agent-6"


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
4. History tables keep versions. When the history table is the one you read from, add its current_sql for the
   present, or its as_of_sql for a past date. For a single day ("as of 3 May"), {{as_of}} is that day's
   sql_end_of_day from "dates" (the state at the end of the day); a snapshot's date column is compared with sql_at.
5. To join, copy one of that join's "patterns" (s = the table you join from, j = the table you join into; rename the
   aliases to yours): "now" for the present, "as_of_date" for a given date, "at_time_of_each_row" for the state at the
   moment of each row of the other table (replace s.{{event_date}} with that row's date column), "latest" or "on_date"
   for snapshots, "latest_version" for tables of dated versions, "lookup" when the table has one row per key.
6. The state of something at the moment of an event comes from its history table with "at_time_of_each_row", not from
   a copy of that value stored on the event row.
6b. What was assigned, started, joined, created, opened, activated or signed up IN a period means versions that
   BEGAN in that period: use the table's started_in_sql (or the "started_in_period" pattern) with the period's start
   and end. What someone had, held or was in DURING a period means versions valid at some point in it (overlap).
7. Snapshot tables repeat every entity once per snapshot date: use one date. Never add up or count across dates.
8. Count rows (COUNT(*)) when each row is one of the things asked about (an event, a request, an order). Count
   distinct keys (COUNT(DISTINCT key)) when the same thing can appear on several rows (versions, snapshot copies,
   or rows multiplied by a join), and when asked how many entities (customers, patients, accounts) there are.
9. A date period from "dates" is start <= column < end, with the date literals exactly as given.
9b. When the question splits or compares by a time unit ("monthly", "per month", "month-over-month", "by week",
   "each quarter"), return one row per period: a column for the period (the date truncated to that unit, e.g.
   TRUNC(date, 'MM')) and one column per measure. Put each change on the later period's row with LAG over the period
   column. Never put several periods side by side in one row.
10. Join only along the joins given. Select only what the question asks for, with readable aliases; order the rows.
10a. A table may list records "left_out_by_default" (test, dummy, internal records found in the data). Unless the
   question asks about such records, add each one's exclusion_sql for every table you read, with that table's alias.
10b. When the question splits a group into categories that come from a joined table ("how many have A and how many
   have B"), write LEFT JOIN with the same ON conditions, so records with no match stay in as their own group (an
   empty category) instead of silently disappearing.
10c. When the question asks which one is highest, lowest, most, fewest, largest, smallest, top or bottom, return
   every group with its figure, sorted so the asked-for extreme comes first (descending for highest/most, ascending
   for lowest/fewest). Do not cut the result to one row (no ROWNUM, FETCH FIRST, LIMIT or rank filter) unless the
   question asks for a number of them ("top 3"): the full ranking is the evidence and shows ties.
{dialect_rules}"""


def _date_sql(adapter: SourceAdapter, iso: str) -> str:
    return adapter.date_literal(date.fromisoformat(iso))


def build_context(extracted: dict, pack: dict, adapter: SourceAdapter) -> dict[str, Any]:
    dates = []
    for d in extracted["dates"]:
        entry = {"text": d["text"], "reading": d["reading"], "kind": d["kind"]}
        if d["kind"] == "point":
            day = date.fromisoformat(d["at"])
            entry["sql_at"] = adapter.date_literal(day)
            # "as of a day" is the state at the end of that day, after everything that happened on it
            entry["sql_end_of_day"] = adapter.datetime_literal(datetime(day.year, day.month, day.day, 23, 59, 59))
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


def _layer_key(index: LayerIndex, table: exp.Table) -> str | None:
    name = f"{table.db}.{table.name}".upper() if table.db else table.name.upper()
    return next((k for k in index.tables if k.upper() == name or k.split(".")[1] == name), None)


def _validity_columns(index: LayerIndex, key: str, dialect: str) -> set[str]:
    """Columns that pick one row per key: a history table's start and end, a snapshot's or version's date."""
    t = index.tables[key]
    grain = [c.upper() for c in (t["grain"] or {}).get("columns", [])]
    if t["kind"] == "history":
        cols = set(grain[1:])
        if t["current_sql"]:
            rule = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {t['current_sql']}", read=dialect)
            cols |= {c.name.upper() for c in rule.find_all(exp.Column)}
        return cols
    if t["kind"] in ("snapshot", "entity_by_date"):
        return set(grain[-1:])
    return set()


def version_rule_issues(sql: str, index: LayerIndex, dialect: str) -> list[str]:
    """Tables with several rows per key that the query reads without the rule picking one row per key.

    A history or snapshot table must use its validity or date column wherever it is read; a table of
    dated versions only when it is joined into (as the main table its rows may be the events counted).
    """
    issues = []
    tree = sqlglot.parse_one(sql, read=dialect)
    for scope in traverse_scope(tree):
        tables = [(alias, src) for alias, src in scope.sources.items() if isinstance(src, exp.Table)]
        main = scope.expression.args.get("from_") if isinstance(scope.expression, exp.Select) else None
        main_alias = main.this.alias_or_name.upper() if main and isinstance(main.this, exp.Table) else None
        for alias, src in tables:
            key = _layer_key(index, src)
            if key is None:
                continue
            kind = index.tables[key]["kind"]
            if kind not in ("history", "snapshot", "entity_by_date"):
                continue
            if kind == "entity_by_date" and alias.upper() == main_alias:
                continue
            need = _validity_columns(index, key, dialect)
            used = {
                c.name.upper()
                for c in scope.columns
                if c.table.upper() == alias.upper() or (not c.table and len(tables) == 1)
            }
            if need and not used & need:
                issues.append(
                    f"{key} is a {kind} table with several rows per key, read without a rule that keeps one "
                    f"(none of {sorted(need)} is used): rows of every version or copy are counted"
                )
    return sorted(set(issues))


def _query_tables(tree: exp.Expression, index: LayerIndex) -> list[str]:
    return sorted({k for t in tree.find_all(exp.Table) if (k := _layer_key(index, t))})


def value_word_issues(sql: str, index: LayerIndex, extracted: dict, dialect: str) -> list[str]:
    """Words of the question that are exactly a stored value of a code column the query reads, but unused.

    "overdue" in a question, OVERDUE among the values of a status column: a query that never
    filters on OVERDUE answers a different question. A word is satisfied when any of the
    values it matches appears in the query.
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    literals = {fold(str(l.this)).strip() for l in tree.find_all(exp.Literal) if l.is_string}
    values: dict[str, list[tuple[str, str, str]]] = {}  # folded value -> (table, column, value)
    for key in _query_tables(tree, index):
        for c in index.tables[key]["columns"]:
            if c.get("shape") not in ("code", "flag") or c.get("sensitive"):
                continue
            for v in c.get("values") or []:
                if isinstance(v, str):
                    values.setdefault(fold(v).strip(), []).append((key, c["name"], v))
    phrases = [p for slot in ("subjects", "conditions", "values") for p in extracted.get(slot, [])]
    issues = []
    for word in sorted({w for p in phrases for w in re.findall(r"[^\W_]+", fold(p)) if len(w) >= 3}):
        hits = [h for form in {word, _stem(word), word + "s"} for h in values.get(form, [])]
        if hits and not any(fold(v).strip() in literals for _, _, v in hits):
            where = "; ".join(sorted({f"{t}.{c} holds '{v}'" for t, c, v in hits}))
            issues.append(f"the question says '{word}', which is a stored value ({where}), but the query never uses it")
    return issues


_COUNTED = re.compile(r"\b(?:how many|number of|count(?: of)?|total(?: number of)?)\s+(?:the\s+|all\s+|unique\s+|distinct\s+)?([^\W\d_]+)")


def count_grain_issues(sql: str, index: LayerIndex, extracted: dict, dialect: str) -> list[str]:
    """COUNT(DISTINCT id) over a table where each row is one event, when the question counts the events.

    A dated-events table (one row per entity per date) records one thing per row; an id on
    it can repeat. Counting distinct ids then counts entities, not the events asked about.
    It stays quiet when the question counts things the id stands for ("how many customers").
    """
    if extracted.get("output") != "count":
        return []
    counted = [m.group(1) for m in _COUNTED.finditer(fold(extracted["question"]))]
    issues = []
    tree = sqlglot.parse_one(sql, read=dialect)
    for scope in traverse_scope(tree):
        if not isinstance(scope.expression, exp.Select):
            continue
        main = scope.expression.args.get("from_")
        if not main or not isinstance(main.this, exp.Table):
            continue
        key = _layer_key(index, main.this)
        if key is None or index.tables[key]["kind"] != "entity_by_date":
            continue
        alias = main.this.alias_or_name.upper()
        single = len([s for s in scope.sources.values() if isinstance(s, exp.Table)]) == 1
        for count in scope.expression.find_all(exp.Count):
            if not isinstance(count.this, exp.Distinct) or count.find_ancestor(exp.Select) is not scope.expression:
                continue
            for col in count.this.find_all(exp.Column):
                if col.table.upper() != alias and not (single and not col.table):
                    continue
                id_words = tokens(col.name)
                if any(_match_strength(t, w) for w in counted for t in id_words):
                    continue  # the question counts what this id stands for
                issues.append(
                    f"COUNT(DISTINCT {col.name}) counts distinct {col.name} values, but each row of {key} is one "
                    f"event and a {col.name} can have several; the question counts "
                    f"{', '.join(repr(w) for w in counted) or 'the events'}, so count rows (COUNT(*))"
                )
    return sorted(set(issues))


_STARTED = re.compile(
    r"\b(assigned|started|began|joined|created|opened|activated|signed up|enrolled|registered|subscribed|onboarded)\b"
)


_HIGH = r"highest|most|largest|biggest|greatest|maximum|top"
_LOW = r"lowest|fewest|least|smallest|minimum|bottom"
_RANKED = re.compile(rf"\b(?<!at )({_HIGH}|{_LOW}|best|worst)\b")
_HOW_MANY = re.compile(
    rf"\b(?:(?:top|bottom|first|last)\s+(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
    rf"|(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:{_HIGH}|{_LOW}|best|worst))\b"
)


def ranking_issues(sql: str, extracted: dict, dialect: str) -> list[str]:
    """A "which one is highest/lowest" question answered with only the winner, without its figure, or unsorted.

    The ranking of every group, with the figure it is ranked by, is the evidence for the answer
    and shows ties; cutting it to one row hides both. A question that asks for a number of
    items ("top 3") may cut, and a question with no groups (a single highest value) is left alone.
    """
    question = fold(extracted["question"])
    found = _RANKED.findall(question)
    if not found or _HOW_MANY.search(question):
        return []
    tree = sqlglot.parse_one(sql, read=dialect)
    if not tree.find(exp.Group):
        return []
    issues = []
    if (
        tree.find(exp.Limit, exp.Fetch)
        or any(c.name.upper() == "ROWNUM" for c in tree.find_all(exp.Column))
        or any(
            isinstance(side, exp.Subquery) and side.find(exp.Max, exp.Min)
            for eq in tree.find_all(exp.EQ)
            for side in (eq.left, eq.right)
        )
        or any(w.this.key in ("rownumber", "rank", "denserank") for w in tree.find_all(exp.Window))
    ):
        issues.append(
            "the question asks which one is highest or lowest, but the query keeps only the top row; return every "
            "group with its figure, sorted, so the ranking and any ties are visible"
        )
    outer = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    figures = {
        p.alias.upper()
        for select in tree.find_all(exp.Select)
        for p in select.expressions
        if isinstance(p, exp.Alias) and p.this.find(exp.AggFunc)
    }
    if outer is not None and not any(
        p.find(exp.AggFunc) or p.alias_or_name.upper() in figures for p in outer.expressions
    ):
        issues.append("return the figure each group is ranked by, not only the group's name")
    order = outer.args.get("order") if outer is not None else None
    if order is None or not order.expressions:
        issues.append("sort the rows so the asked-for extreme comes first")
    else:
        first = order.expressions[0]
        descending = bool(first.args.get("desc"))
        high = any(re.fullmatch(_HIGH, w) for w in found)
        low = any(re.fullmatch(_LOW, w) for w in found)
        if high and not low and not descending:
            issues.append("the question asks for the highest, so sort by the figure in descending order")
        elif low and not high and descending:
            issues.append("the question asks for the lowest, so sort by the figure in ascending order")
    return issues


def started_in_issues(sql: str, index: LayerIndex, extracted: dict, dialect: str) -> list[str]:
    """A question about what began in a period, answered with versions merely valid during it.

    "Accounts assigned to a plan in June" are the versions whose start date is in June. A query
    that keeps every version overlapping June (start before the end, end after the start) also
    counts everyone assigned long before. Flags a history table read with its end date but no
    lower bound on its start date.
    """
    words = _STARTED.findall(fold(extracted["question"]))
    if not words or not any(d["kind"] == "period" for d in extracted["dates"]):
        return []
    issues = []
    tree = sqlglot.parse_one(sql, read=dialect)
    for scope in traverse_scope(tree):
        tables = [(alias, src) for alias, src in scope.sources.items() if isinstance(src, exp.Table)]
        for alias, src in tables:
            key = _layer_key(index, src)
            if key is None or index.tables[key]["kind"] != "history":
                continue
            grain = [c.upper() for c in (index.tables[key]["grain"] or {}).get("columns", [])]
            if len(grain) < 2:
                continue
            start = grain[1]
            end_cols = _validity_columns(index, key, dialect) - {start}

            def mine(node) -> bool:
                return isinstance(node, exp.Column) and (
                    node.table.upper() == alias.upper() or (not node.table and len(tables) == 1)
                )

            lower_bound = any(
                (isinstance(n, (exp.GTE, exp.GT)) and mine(n.left) and n.left.name.upper() == start)
                or (isinstance(n, (exp.LTE, exp.LT)) and mine(n.right) and n.right.name.upper() == start)
                or (isinstance(n, exp.Between) and mine(n.this) and n.this.name.upper() == start)
                for n in scope.expression.find_all(exp.GTE, exp.GT, exp.LTE, exp.LT, exp.Between)
            )
            uses_end = any(mine(c) and c.name.upper() in end_cols for c in scope.columns)
            if uses_end and not lower_bound:
                issues.append(
                    f"the question asks what was {words[0]} in a period, but the query keeps every {key} version "
                    f"valid during it, including ones that began earlier; filter by when it began: "
                    f"{start} >= period start AND {start} < period end"
                )
    return sorted(set(issues))


def unmatched_rows(adapter: SourceAdapter, index: LayerIndex, sql: str, dialect: str) -> list[dict[str, Any]]:
    """Records an inner join into a history table silently leaves out.

    Each entry says how many, which table, whether the result is split by that table's values
    (then the missing records are a whole category that vanished), and a note in plain words.
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(tree, exp.Select) or tree.args.get("with"):
        return []
    found = []
    joins = tree.args.get("joins") or []
    for i, join in enumerate(joins):
        if join.side or not isinstance(join.this, exp.Table):
            continue
        key = _layer_key(index, join.this)
        if key is None or index.tables[key]["kind"] != "history" or not join.args.get("on"):
            continue
        alias = join.this.alias_or_name.upper()
        cols = [c for c in join.args["on"].find_all(exp.Column) if c.table.upper() == alias]
        if not cols:
            continue
        probe = tree.copy()
        for k in ("group", "order", "having", "limit", "distinct"):
            probe.set(k, None)
        for later in probe.args["joins"][i:]:  # keep the unmatched rows through the joins that follow
            later.set("side", "LEFT")
        probe.set("expressions", [exp.Count(this=exp.Star())])
        probe = probe.where(exp.Is(this=cols[0].copy(), expression=exp.Null()))
        missing = int(adapter.run_select(probe.sql(dialect=dialect), max_rows=1).rows[0][0] or 0)
        if not missing:
            continue
        # is the result split by this table's values (selected or grouped by)?
        shown = [*tree.expressions, *(tree.args["group"].expressions if tree.args.get("group") else [])]
        category = any(c.table.upper() == alias for e in shown for c in e.find_all(exp.Column))
        found.append(
            {
                "count": missing,
                "table": key,
                "category": category,
                "technical": f"{missing} rows found no matching row in {key} for that join and are not included",
                "plain": f"{missing:,} of the records counted had no matching entry at that point in time and are not "
                "included",
            }
        )
    return found


def left_out_issues(sql: str, index: LayerIndex, extracted: dict, dialect: str) -> list[str]:
    """Tables read without applying the records they leave out by default, when the question does not ask for them.

    A left-out term counts as applied when every value it compares against appears in the query; a
    question that names the term's own words (for example asks about test records) needs no exclusion.
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    query_literals = {str(l.this) for l in tree.find_all(exp.Literal) if l.is_string}
    asked = set(tokens(extracted["question"]))
    issues = []
    for key in _query_tables(tree, index):
        for term in index.terms.values():
            if term["table"] != key or not term.get("left_out_by_default") or not term.get("exclusion_sql"):
                continue
            if asked & set(tokens(term["term"])):
                continue  # the question is about these records
            term_tree = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {term['sql']}", read=dialect)
            needed = {str(l.this) for l in term_tree.find_all(exp.Literal) if l.is_string}
            if needed and not needed <= query_literals:
                issues.append(
                    f"{key} normally leaves out '{term['term']}' ({term['definition']}); the question does not ask "
                    f"for them, so add: {term['exclusion_sql']}. Run the same query again if they should be counted"
                )
    return sorted(set(issues))


def dropped_category_issues(adapter: SourceAdapter, index: LayerIndex, sql: str, dialect: str) -> list[str]:
    """A result split by a joined table's values while an inner join silently drops the records without one."""
    return [
        f"{u['count']} records have no matching entry in {u['table']}; the inner join drops them while the result is "
        f"split by that table's values, so they vanish from the split. Use LEFT JOIN with the same ON conditions to "
        f"keep them as their own group, or run the same query again if they should be left out"
        for u in unmatched_rows(adapter, index, sql, dialect)
        if u["category"]
    ]


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
    warned: set[str] = set()  # queries already sent back once; the same query again means "intended"

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
            dialect = adapter.sqlglot_dialect
            soft = (
                []
                if problems or sql in warned
                else version_rule_issues(sql, index, dialect)
                + value_word_issues(sql, index, extracted, dialect)
                + count_grain_issues(sql, index, extracted, dialect)
                + started_in_issues(sql, index, extracted, dialect)
                + ranking_issues(sql, extracted, dialect)
                + left_out_issues(sql, index, extracted, dialect)
            )
            if not problems and sql not in warned:
                try:  # needs the database: how many records an inner join would silently drop
                    soft += dropped_category_issues(adapter, index, sql, dialect)
                except Exception:
                    pass  # if it cannot be measured, the query is not held back for it
            if problems:
                observation = {"error": "the query was not run", "problems": problems}
            elif soft:
                warned.add(sql)
                observation = {
                    "warning": "the query was not run: it may not answer the question as asked",
                    "problems": soft,
                    "hint": "fix the query using the context: join patterns, stored values, counting rules. "
                    "If the query is right as it is, run the same query again unchanged.",
                }
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
                        try:
                            dropped = unmatched_rows(adapter, index, sql, adapter.sqlglot_dialect)
                            answer["notes"] = [u["plain"] for u in dropped]
                            answer["technical_notes"] = [u["technical"] for u in dropped]
                        except Exception as exc:  # a note we could not compute is not worth failing the answer
                            answer["notes"] = []
                            answer["technical_notes"] = [f"could not check for unmatched rows: {adapter.map_error(exc)}"]
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
