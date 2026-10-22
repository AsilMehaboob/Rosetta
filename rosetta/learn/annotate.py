"""Annotate: propose meanings.

Builds one evidence pack per table from the stored profile, findings and links
(never the source database), asks the LLM to describe the table, its columns
and code values, and to propose business terms as SQL. Code then checks every
claim against the evidence: names must exist, quoted values must have been
seen, and SQL must parse and touch only this table and the tables it links to.
A second pass groups terms that mean the same thing and flags a word defined
two ways. Everything here is a proposal; Assemble tests each term on the data.
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from rosetta.learn.detect import FINDINGS, LINKS
from rosetta.learn.profile import PROFILES, RUNS
from rosetta.llm import JsonLLM, QuotaExhausted
from rosetta.storage import Storage

log = logging.getLogger(__name__)

ANNOTATIONS = "annotations"
TERMS = "terms"
PROMPT_VERSION = "annotate-3"
MERGE_PROMPT_VERSION = "merge-1"


@dataclass(frozen=True)
class AnnotateConfig:
    code_values_max: int = 50  # list every value of a code column up to this many
    example_values: int = 5  # otherwise show this many of the most frequent values
    comment_max_chars: int = 400
    incoming_links_max: int = 30

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


_CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low", "unknown"]}

TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "row_meaning": {"type": "string"},
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": _CONFIDENCE,
                },
                "required": ["name", "description", "confidence"],
            },
        },
        "value_meanings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "value": {"type": "string"},
                    "meaning": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": _CONFIDENCE,
                },
                "required": ["column", "value", "meaning", "evidence", "confidence"],
            },
        },
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "synonyms": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": ["filter", "measure", "dimension"]},
                    "definition": {"type": "string"},
                    "sql": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": _CONFIDENCE,
                },
                "required": ["term", "synonyms", "kind", "definition", "sql", "evidence", "confidence"],
            },
        },
        "unknowns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "row_meaning", "columns", "value_meanings", "terms", "unknowns"],
}

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "term_ids": {"type": "array", "items": {"type": "string"}},
                    "same_meaning": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["canonical", "term_ids", "same_meaning", "note"],
            },
        }
    },
    "required": ["groups"],
}

SYSTEM = """You are a data analyst learning an unfamiliar business database that has no documentation.
You get evidence about one table, measured from the data itself:
- structure: what one row stands for, and whether the table keeps history (versions with start and end
  dates) or repeated snapshots (the same entities copied once per date), with the rule for the current row
  and for reading a past date;
- columns: type, share of rows filled, number of distinct values, observed values with row counts,
  and the database comment if there is one. Comments are hints only and can be wrong or copied from
  another column; when a comment disagrees with the values, trust the values and say so;
- current_rows: for code columns of history or snapshot tables, how many rows of each value are current;
- links to other tables, and the rows of small reference tables.

Rules:
1. Use only table names, column names and values that appear in the evidence. Never invent any.
2. Every claim must rest on the evidence. In "evidence", cite the values and numbers that support it.
3. When the evidence does not settle a meaning, give confidence "unknown" instead of guessing.
4. Describe every column, in the order given.
5. value_meanings: explain the values of code and flag columns.
6. terms: propose the words a business user would say about this table: states (what makes a row
   active, open, valid, available, current), groups, events and counts. For each write SQL in the
   {dialect} dialect. {dialect_rules}
   - filter: only a boolean condition on this table's rows (what follows WHERE), column names unqualified.
   - measure: only an aggregate expression such as COUNT(DISTINCT X), never a SELECT statement.
     When one row is not one entity, count distinct entities.
   - dimension: only a column or expression to group by.
   A filter may test another table only through the links given, as
   COLUMN IN (SELECT LINKED_COLUMN FROM SCHEMA.TABLE WHERE ...).
   When a linked table's rows carry flags or groupings (for example an indicator column that splits
   the codes this table uses into two sets), propose terms that use them through the link.
7. If the table keeps history or snapshots, a term about the present must include structure.current_sql,
   copied exactly as given (it holds the exact stored marker, time of day included); never write your own
   version of it. A term about a past moment uses structure.as_of_sql with {{as_of}} left in place.
   The same applies to a linked table's current_sql inside a subquery. Say so in the definition.
8. Copy values exactly as shown, including case, spaces and accents.
9. List in "unknowns" what you could not work out."""

MERGE_SYSTEM = """You are given business terms proposed separately for each table of one database.
Group terms that a business user would mean by the same words. For each group give a canonical name,
the term ids, whether they truly mean the same thing (same_meaning), and a short note. When the same
word is defined differently in different places, put those terms in one group with same_meaning false
and explain the difference. Use only the term ids given; every id appears in exactly one group."""


def _clean_comment(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    text = re.sub(r"_x[0-9A-Fa-f]{4}_", " ", text)  # spreadsheet escapes such as _x000D_
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] or None


def _best_links(links: list[dict]) -> list[dict]:
    return [l for l in links if l["status"] == "accepted" and l["best"]]


def build_pack(
    key: str,
    profiles: dict[str, dict],
    findings: dict[str, dict],
    links: list[dict],
    cfg: AnnotateConfig,
) -> dict[str, Any]:
    """Everything the LLM may see about one table, in a fixed order."""
    profile, finding = profiles[key], findings[key]
    behaviour = finding.get("code_behaviour") or {}
    detail = finding.get("history") or finding.get("snapshot") or finding.get("versioned") or {}

    columns = []
    for c in sorted(profile["columns"], key=lambda c: c["position"]):
        shape = finding["columns"][c["name"]]["shape"]
        entry: dict[str, Any] = {
            "name": c["name"],
            "data_type": c["data_type"],
            "shape": shape,
            "filled_share": finding["columns"][c["name"]]["filled_share"],
            "distinct": c["distinct_count"],
            "comment": _clean_comment(c["comment"], cfg.comment_max_chars),
        }
        if c["sensitive"]:
            entry["values_hidden"] = "personal or identifying data"
        else:
            if c["min"] is not None and shape in ("date", "number", "unique"):
                entry["range"] = [c["min"], c["max"]]
            if shape in ("code", "flag", "constant"):
                entry["values"] = [[v["value"], v["count"]] for v in c["values"][: cfg.code_values_max]]
                entry["values_complete"] = c["values_complete"] and c["distinct_count"] <= cfg.code_values_max
            elif c["values"]:
                entry["examples"] = [v["value"] for v in c["values"][: cfg.example_values]]
            if c["name"] in behaviour:
                entry["current_rows"] = [[v["value"], v["rows"], v["current_rows"]] for v in behaviour[c["name"]]]
        columns.append(entry)

    best = _best_links(links)
    outgoing = sorted(
        ({"column": l["child"]["column"], "to": f"{l['parent']['table']}.{l['parent']['column']}", "notes": l["notes"]}
         for l in best if l["child"]["table"] == key),
        key=lambda l: (l["column"], l["to"]),
    )
    incoming = sorted(
        (f"{l['child']['table']}.{l['child']['column']} -> {l['parent']['column']}"
         for l in best if l["parent"]["table"] == key)
    )[: cfg.incoming_links_max]

    linked = {}
    for target in sorted({l["to"].rsplit(".", 1)[0] for l in outgoing}):
        other = findings[target]
        other_detail = other.get("history") or other.get("snapshot") or other.get("versioned") or {}
        linked[target] = {
            "kind": other["kind"],
            "grain": other["grain"]["text"] if other["grain"] else None,
            "rule": other_detail.get("rule_text"),
            "current_sql": other_detail.get("current_sql"),
            "columns": [c["name"] for c in sorted(profiles[target]["columns"], key=lambda c: c["position"])],
            "rows": other.get("lookup_rows"),
        }

    return {
        "table": key,
        "row_count": finding["row_count"],
        "structure": {
            "kind": finding["kind"],
            "grain": finding["grain"]["text"] if finding["grain"] else None,
            "rule": detail.get("rule_text"),
            "current_sql": detail.get("current_sql"),
            "as_of_sql": detail.get("as_of_sql"),
        },
        "columns": columns,
        "rows": finding.get("lookup_rows"),
        "links_out": outgoing,
        "links_in": incoming,
        "linked_tables": linked,
    }


def _table_names(key: str) -> set[str]:
    schema, table = key.split(".", 1)
    return {table.upper(), f"{schema}.{table}".upper()}


def check_sql(
    sql: str, kind: str, key: str, allowed: dict[str, dict], dialect: str
) -> list[str]:
    """Problems with a proposed term's SQL; empty when it only uses what exists.

    `allowed` maps table keys (this table and the tables it links to) to their
    profiles, so every table, column and quoted value can be checked.
    """
    if ";" in sql:
        return ["contains a statement separator"]
    sql = sql.replace("{as_of}", "DATE '2000-01-01'")  # the as-of placeholder is filled in at answer time
    schema, table = key.split(".", 1)
    wrapper = {
        "filter": f"SELECT 1 FROM {schema}.{table} WHERE {sql}",
        "measure": f"SELECT {sql} FROM {schema}.{table}",
        "dimension": f"SELECT {sql} FROM {schema}.{table}",
    }[kind]
    try:
        statements = sqlglot.parse(wrapper, read=dialect)
    except sqlglot.errors.ParseError as exc:
        return [f"does not parse: {str(exc).splitlines()[0]}"]
    if len(statements) != 1 or statements[0] is None:
        return ["is not a single expression"]
    tree = statements[0]

    by_name = {}
    for k in allowed:
        for name in _table_names(k):
            by_name[name] = k
    issues, columns, checked = [], {}, set()

    def scope_columns(scope) -> dict[str, dict]:
        """Columns of a scope's own tables. A subquery reading the outer table's column
        is legal SQL but almost always a mistake, so it is not allowed."""
        found = {}
        for source in scope.sources.values():
            if not isinstance(source, exp.Table):
                continue
            name = f"{source.db}.{source.name}".upper() if source.db else source.name.upper()
            if name not in by_name:
                issues.append(f"uses table {name}, which is not this table or a linked one")
                continue
            for c in allowed[by_name[name]]["columns"]:
                found.setdefault(c["name"].upper(), c)
        return found

    for scope in traverse_scope(tree):  # innermost first; an outer scope repeats inner columns
        readable = scope_columns(scope)
        for col in scope.columns:
            if id(col) in checked:
                continue
            checked.add(id(col))
            profile = readable.get(col.name.upper())
            if profile is None:
                issues.append(f"uses column {col.name}, which its table does not have")
            else:
                columns[id(col)] = profile

    # a quoted value compared with a column must be one of that column's values
    def seen(column: exp.Expression, literal: exp.Expression) -> None:
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal):
            return
        profile = columns.get(id(column))
        if not profile or not profile["values_complete"] or profile["sensitive"]:
            return
        values = {str(v["value"]) for v in profile["values"]}
        if literal.this not in values:
            issues.append(f"compares {column.name} with {literal.this!r}, which never occurs in it")

    for node in tree.find_all(exp.EQ, exp.NEQ):
        seen(node.left, node.right)
        seen(node.right, node.left)
    for node in tree.find_all(exp.In):
        for item in node.expressions:
            seen(node.this, item)
    return sorted(set(issues))


def _term_id(key: str, term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    return f"{key}:{slug}"


def validate(
    raw: dict[str, Any], key: str, profiles: dict[str, dict], allowed_keys: set[str], dialect: str
) -> tuple[dict[str, Any], list[str]]:
    """Keep what the evidence supports; record everything dropped or doubtful."""
    profile = profiles[key]
    names = [c["name"] for c in sorted(profile["columns"], key=lambda c: c["position"])]
    by_col = {c["name"].upper(): c for c in profile["columns"]}
    issues = []

    described = {}
    for c in raw.get("columns", []):
        name = c["name"].upper()
        if name not in by_col:
            issues.append(f"described column {c['name']}, which does not exist")
        else:
            described.setdefault(by_col[name]["name"], c)
    columns = []
    for name in names:
        c = described.get(name)
        if c is None:
            issues.append(f"column {name} not described")
            columns.append({"name": name, "description": "", "confidence": "unknown"})
        else:
            columns.append({"name": name, "description": c["description"], "confidence": c["confidence"]})

    meanings = []
    for m in raw.get("value_meanings", []):
        col = by_col.get(m["column"].upper())
        if col is None:
            issues.append(f"value meaning for column {m['column']}, which does not exist")
            continue
        if col["values_complete"] and m["value"] not in {str(v["value"]) for v in col["values"]}:
            issues.append(f"value meaning for {col['name']} = {m['value']!r}, which never occurs")
            continue
        meanings.append({**m, "column": col["name"]})
    meanings.sort(key=lambda m: (m["column"], m["value"]))

    allowed = {k: profiles[k] for k in allowed_keys}
    terms, seen_ids = [], set()
    for t in raw.get("terms", []):
        term_id = _term_id(key, t["term"])
        if term_id in seen_ids:
            issues.append(f"term {t['term']!r} proposed twice; kept the first")
            continue
        seen_ids.add(term_id)
        problems = check_sql(t["sql"], t["kind"], key, allowed, dialect)
        terms.append(
            {
                "id": term_id,
                "table": key,
                **t,
                "synonyms": sorted(set(t["synonyms"])),
                "status": "invalid" if problems else "proposed",
                "issues": problems,
            }
        )
    terms.sort(key=lambda t: t["id"])

    annotation = {
        "description": raw.get("description", ""),
        "row_meaning": raw.get("row_meaning", ""),
        "columns": columns,
        "value_meanings": meanings,
        "terms": terms,
        "unknowns": sorted(raw.get("unknowns", [])),
    }
    return annotation, sorted(issues)


def annotate_table(
    llm: JsonLLM,
    key: str,
    profiles: dict[str, dict],
    findings: dict[str, dict],
    links: list[dict],
    dialect: str,
    dialect_rules: str,
    cfg: AnnotateConfig,
) -> dict[str, Any]:
    pack = build_pack(key, profiles, findings, links, cfg)
    prompt = json.dumps(pack, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    system = SYSTEM.format(dialect=dialect, dialect_rules=dialect_rules)
    raw = llm.generate_json(system=system, prompt=prompt, schema=TABLE_SCHEMA, prompt_version=PROMPT_VERSION)
    allowed = {key} | set(pack["linked_tables"])
    annotation, issues = validate(raw, key, profiles, allowed, dialect)
    return {
        "table": key,
        "model": llm.model,
        "prompt_version": PROMPT_VERSION,
        "input_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "annotation": annotation,
        "issues": issues,
    }


def merge_terms(llm: JsonLLM, terms: list[dict]) -> tuple[list[dict], list[str]]:
    """Group same-meaning terms across tables and flag words defined two ways."""
    if not terms:
        return [], []
    listing = [
        {"id": t["id"], "term": t["term"], "synonyms": t["synonyms"], "kind": t["kind"], "definition": t["definition"]}
        for t in sorted(terms, key=lambda t: t["id"])
    ]
    raw = llm.generate_json(
        system=MERGE_SYSTEM,
        prompt=json.dumps(listing, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        schema=MERGE_SCHEMA,
        prompt_version=MERGE_PROMPT_VERSION,
    )
    known = {t["id"] for t in terms}
    placed, groups, issues = set(), [], []
    for g in raw.get("groups", []):
        ids = sorted(i for i in set(g["term_ids"]) if i in known and i not in placed)
        unknown = sorted(set(g["term_ids"]) - known)
        if unknown:
            issues.append(f"merge named unknown term ids {unknown}")
        if ids:
            placed |= set(ids)
            groups.append({**g, "term_ids": ids})
    for missing in sorted(known - placed):  # anything the LLM left out stands alone
        term = next(t for t in terms if t["id"] == missing)
        groups.append({"canonical": term["term"], "term_ids": [missing], "same_meaning": True, "note": ""})
    groups.sort(key=lambda g: (g["canonical"].lower(), g["term_ids"]))
    return groups, issues


def run_annotate(
    llm: JsonLLM,
    storage: Storage,
    dialect: str,
    dialect_rules: str,
    cfg: AnnotateConfig | None = None,
) -> dict[str, Any]:
    """Annotate every detected table, then merge the proposed terms."""
    cfg = cfg or AnnotateConfig()
    findings = {k: storage.read(FINDINGS, k) for k in storage.keys(FINDINGS)}
    if not findings:
        raise RuntimeError("No findings stored; run the detect stage first")
    profiles = {k: storage.read(PROFILES, k) for k in findings}
    links = (storage.read(LINKS, "links") or {"links": []})["links"]

    status, all_terms, quota_error = {}, [], None
    for i, key in enumerate(sorted(findings), 1):  # one at a time: free-tier rate limits
        if quota_error:
            status[key] = {"status": "not_run", "reason": quota_error}
            continue
        log.info("annotate %d/%d %s", i, len(findings), key)
        try:
            result = annotate_table(llm, key, profiles, findings, links, dialect, dialect_rules, cfg)
        except QuotaExhausted as exc:  # later tables would only fail too; rerun tomorrow from the cache
            quota_error = str(exc)
            status[key] = {"status": "not_run", "reason": quota_error}
            continue
        except Exception as exc:  # recorded, never hidden
            status[key] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        storage.write(ANNOTATIONS, key, result)
        terms = result["annotation"]["terms"]
        all_terms += terms
        status[key] = {
            "status": "annotated",
            "terms": len(terms),
            "invalid_terms": sum(1 for t in terms if t["status"] == "invalid"),
            "issues": len(result["issues"]),
        }

    usable = [t for t in all_terms if t["status"] == "proposed"]
    if quota_error:
        groups, merge_issues = [], [f"merge not run: {quota_error}"]
    else:
        log.info("merging %d terms", len(usable))
        try:
            groups, merge_issues = merge_terms(llm, usable)
        except Exception as exc:
            groups, merge_issues = [], [f"merge failed: {type(exc).__name__}: {exc}"]
    storage.write(
        TERMS,
        "terms",
        {
            "model": llm.model,
            "prompt_version": PROMPT_VERSION,
            "merge_prompt_version": MERGE_PROMPT_VERSION,
            "terms": sorted(all_terms, key=lambda t: t["id"]),
            "groups": groups,
            "issues": merge_issues,
        },
    )

    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "stage": "annotate",
        "model": llm.model,
        "prompt_version": PROMPT_VERSION,
        "config": asdict(cfg),
        "config_digest": cfg.digest(),
        "llm_calls": llm.calls,
        "llm_cached": llm.cached,
        "tables": dict(sorted(status.items())),
        "terms": {
            "proposed": len(usable),
            "invalid": sum(1 for t in all_terms if t["status"] == "invalid"),
            "groups": len(groups),
            "conflicts": sum(1 for g in groups if not g["same_meaning"]),
        },
    }
    storage.write(RUNS, f"annotate-{run['run_id']}", run)
    return run
