"""Assemble: test and publish.

Takes the proposed terms, carries filters on small reference tables through
their links, applies hand corrections from the overrides file, and runs every
term on the source database. A filter must match some rows but not all of
them; a measure or dimension must run; a measure that counts rows where a row
is a version or a copy, not an entity, is held back. Terms that pass are
published with their counts as proof, the rest are kept with the reason.

The layer is hashed into a version and published atomically. Each version is
also exported as YAML with a changelog, so a reviewer (and git) can see what
was learned and every hand correction.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import sqlglot
import yaml
from sqlglot import exp

from rosetta.adapters.base import SourceAdapter
from rosetta.learn.annotate import ANNOTATIONS, TERMS, check_sql
from rosetta.learn.detect import FINDINGS, LINKS, _as_datetime
from rosetta.learn.profile import PROFILES, RUNS
from rosetta.storage import Storage

LAYERS = "layers"
VERIFICATIONS = "verifications"


@dataclass(frozen=True)
class AssembleConfig:
    version: str = "assemble-1"
    derive_max_parent_rows: int = 50  # filters on reference tables this small are carried through links
    column_values_max: int = 50  # code values listed in the layer
    timeout_seconds: int = 60
    workers: int = 4
    overrides_path: str = "semantic/overrides.yaml"
    export_dir: str = "semantic"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _src(adapter: SourceAdapter, key: str) -> str:
    schema, table = key.split(".", 1)
    return f"{adapter.quote(schema)}.{adapter.quote(table)}"


def _detail(finding: dict) -> dict:
    return finding.get("history") or finding.get("snapshot") or finding.get("versioned") or {}


def derive_linked_terms(
    terms: list[dict], findings: dict[str, dict], links: list[dict], adapter: SourceAdapter, cfg: AssembleConfig
) -> list[dict]:
    """A filter on a small reference table also selects rows of every table that links to it."""
    by_table: dict[str, list[dict]] = {}
    for t in terms:
        if t["kind"] == "filter":
            by_table.setdefault(t["table"], []).append(t)
    derived = []
    for link in links:
        if link["status"] != "accepted" or not link["best"]:
            continue
        parent, child = link["parent"]["table"], link["child"]["table"]
        info = findings[parent]
        if info["kind"] != "keyed" or info["row_count"] > cfg.derive_max_parent_rows:
            continue
        for t in by_table.get(parent, []):
            column, key = link["child"]["column"], link["parent"]["column"]
            name = f"{t['term']} (via {column})"
            derived.append(
                {
                    "id": f"{child}:{_slug(name)}",
                    "table": child,
                    "term": name,
                    "synonyms": t["synonyms"],
                    "kind": "filter",
                    "definition": f"rows whose {column} is one of the {parent} rows that are '{t['term']}': "
                    f"{t['definition']}",
                    "sql": f"{adapter.quote(column)} IN (SELECT {adapter.quote(key)} FROM {_src(adapter, parent)} "
                    f"WHERE {t['sql']})",
                    "evidence": f"{t['evidence']}; {column} links to {parent}.{key} "
                    f"({round(link['containment'] * 100, 1)}% of values match)",
                    "confidence": t["confidence"],
                    "origin": {"type": "derived", "from": t["id"], "link": f"{child}.{column} -> {parent}.{key}"},
                }
            )
    return sorted(derived, key=lambda t: t["id"])


def load_overrides(path: str) -> list[dict]:
    file = Path(path)
    if not file.exists():
        return []
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    overrides = data.get("overrides") or []
    for i, o in enumerate(overrides):
        if not o.get("reason"):
            raise ValueError(f"override {i + 1} in {path} has no reason; every hand correction must say why")
        if o.get("kind") not in ("term", "column") or o.get("action") not in ("add", "replace", "remove"):
            raise ValueError(f"override {i + 1} in {path}: kind must be term|column, action add|replace|remove")
    return overrides


def apply_term_overrides(terms: list[dict], overrides: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hand corrections on top of the learned terms; the learned output itself is never edited."""
    by_id = {t["id"]: t for t in terms}
    applied = []
    for o in overrides:
        if o["kind"] != "term":
            continue
        if o["action"] == "add":
            fields = o["term"]
            term_id = f"{fields['table']}:{_slug(fields['term'])}"
            new = {"synonyms": [], "evidence": "", "confidence": "high", **fields, "id": term_id}
            new["origin"] = {"type": "override", "reason": o["reason"]}
            applied.append({"action": "add", "id": term_id, "before": by_id.get(term_id), "after": new, "reason": o["reason"]})
            by_id[term_id] = new
        elif o["id"] not in by_id:
            applied.append({"action": o["action"], "id": o["id"], "error": "no such term", "reason": o["reason"]})
        elif o["action"] == "remove":
            applied.append({"action": "remove", "id": o["id"], "before": by_id.pop(o["id"]), "reason": o["reason"]})
        else:
            before = by_id[o["id"]]
            after = {**before, **o["term"], "id": o["id"], "origin": {"type": "override", "reason": o["reason"]}}
            applied.append({"action": "replace", "id": o["id"], "before": before, "after": after, "reason": o["reason"]})
            by_id[o["id"]] = after
    return sorted(by_id.values(), key=lambda t: t["id"]), applied


def _as_of_literal(adapter: SourceAdapter, finding: dict, profile: dict) -> str:
    """A date inside the data to test as-of terms with: the latest start of a version."""
    start = (finding.get("history") or {}).get("start")
    columns = {c["name"]: c for c in profile["columns"]}
    candidates = [columns[start]["max"]] if start else [
        c["max"] for c in profile["columns"] if adapter.type_family(c["data_type"]) == "date"
    ]
    days = sorted(d for d in (_as_datetime(v) for v in candidates) if d and d.year < 3000)
    return adapter.date_literal(days[-1].date() if days else date.today())


def _counts_rows(sql: str, dialect: str) -> bool:
    """True when a measure counts rows (COUNT(*) or COUNT(col)) instead of distinct values."""
    try:
        tree = sqlglot.parse_one(f"SELECT {sql} FROM t", read=dialect)
    except sqlglot.errors.ParseError:
        return False
    return any(not isinstance(node.this, exp.Distinct) for node in tree.find_all(exp.Count))


def _norm(sql: str) -> str:
    return re.sub(r"\s+", "", sql.replace('"', "")).upper()


def verify_term(
    adapter: SourceAdapter, term: dict, findings: dict[str, dict], profiles: dict[str, dict], cfg: AssembleConfig
) -> dict[str, Any]:
    finding, profile = findings[term["table"]], profiles[term["table"]]
    src = _src(adapter, term["table"])
    sql = term["sql"].replace("{as_of}", _as_of_literal(adapter, finding, profile))
    detail = _detail(finding)
    current = detail.get("current_sql")
    scope = (
        "as_of" if "{as_of}" in term["sql"]
        else "current" if current and _norm(current) in _norm(term["sql"])
        else "all_rows"
    )
    test = {
        "filter": f"SELECT COUNT(*) FROM {src} WHERE {sql}",
        "measure": f"SELECT {sql} FROM {src}",
        "dimension": f"SELECT COUNT(DISTINCT {sql}) FROM {src}",
    }[term["kind"]]
    result = {"id": term["id"], "test_sql": test, "scope": scope, "table_rows": finding["row_count"]}
    try:
        value = adapter.run_select(test, timeout=cfg.timeout_seconds, max_rows=1).rows[0][0]
    except Exception as exc:
        return {**result, "status": "unverified", "reason": adapter.map_error(exc)}

    value = float(value) if value is not None and not isinstance(value, (int, float)) else value
    result["value"] = value
    rows = finding["row_count"]
    if term["kind"] == "filter":
        if not value:
            return {**result, "status": "unverified", "reason": "matches no rows"}
        if value == rows:
            return {**result, "status": "unverified", "reason": "matches every row, so it filters nothing"}
        return {**result, "status": "verified", "proof": f"matches {value} of {rows} rows"}
    if value is None:
        return {**result, "status": "unverified", "reason": "returns nothing"}
    # only versions and copies repeat an entity; in an event or request table each row is one thing
    if term["kind"] == "measure" and finding["kind"] in ("history", "snapshot") and _counts_rows(
        term["sql"], adapter.sqlglot_dialect
    ):
        grain = finding["grain"]["text"] if finding["grain"] else "unknown"
        return {
            **result,
            "status": "unverified",
            "reason": f"counts rows, but one row is not one entity here ({grain}); count distinct entities",
        }
    proof = f"returns {value}" if term["kind"] == "measure" else f"{value} distinct groups"
    return {**result, "status": "verified", "proof": proof}


def _table_section(key: str, finding: dict, profile: dict, annotation: dict | None, cfg: AssembleConfig) -> dict:
    annotation = annotation or {}
    described = {c["name"]: c for c in annotation.get("columns", [])}
    meanings: dict[str, dict[str, str]] = {}
    for m in annotation.get("value_meanings", []):
        meanings.setdefault(m["column"], {})[m["value"]] = m["meaning"]
    columns = []
    for c in sorted(profile["columns"], key=lambda c: c["position"]):
        shape = finding["columns"][c["name"]]["shape"]
        entry = {
            "name": c["name"],
            "type": c["data_type"],
            "shape": shape,
            "description": described.get(c["name"], {}).get("description", ""),
            "confidence": described.get(c["name"], {}).get("confidence", "unknown"),
        }
        if c["sensitive"]:
            entry["sensitive"] = True
        elif shape in ("code", "flag", "constant"):
            entry["values"] = [v["value"] for v in c["values"][: cfg.column_values_max]]
        if c["name"] in meanings:
            entry["meanings"] = dict(sorted(meanings[c["name"]].items()))
        columns.append(entry)
    detail = _detail(finding)
    return {
        "description": annotation.get("description", ""),
        "row_meaning": annotation.get("row_meaning", ""),
        "row_count": finding["row_count"],
        "kind": finding["kind"],
        "grain": finding["grain"],
        "rule": detail.get("rule_text"),
        "current_sql": detail.get("current_sql"),
        "as_of_sql": detail.get("as_of_sql"),
        "columns": columns,
        "unknowns": annotation.get("unknowns", []),
    }


def _apply_column_overrides(tables: dict, overrides: list[dict]) -> list[dict]:
    applied = []
    for o in overrides:
        if o["kind"] != "column":
            continue
        table = tables.get(o.get("table"))
        column = next((c for c in table["columns"] if c["name"] == o.get("column")), None) if table else None
        if column is None:
            applied.append({"action": o["action"], "id": f"{o.get('table')}.{o.get('column')}", "error": "no such column", "reason": o["reason"]})
            continue
        before = dict(column)
        column.update(o.get("fields") or {})
        applied.append({"action": "replace", "id": f"{o['table']}.{o['column']}", "before": before, "after": dict(column), "reason": o["reason"]})
    return applied


def _changelog(history: list[dict]) -> str:
    lines = [
        "# Semantic layer changelog",
        "",
        "Generated by the Assemble stage. Every published version is listed, newest first,",
        "with the terms it added, removed or changed and every hand correction (override) applied.",
        "",
    ]
    for h in reversed(history):
        lines += [f"## {h['version']} ({h['published_at']})", ""]
        lines.append(f"- previous version: {h['previous'] or 'none'}")
        lines.append(f"- terms verified: {h['verified']}, held back: {h['unverified']}")
        for label in ("added", "removed", "changed"):
            if h[label]:
                lines.append(f"- {label}: " + ", ".join(h[label]))
        for o in h["overrides"]:
            status = f" (not applied: {o['error']})" if o.get("error") else ""
            lines.append(f"- hand correction, {o['action']} {o['id']}{status}: {o['reason']}")
        lines.append("")
    return "\n".join(lines)


def run_assemble(adapter: SourceAdapter, storage: Storage, cfg: AssembleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AssembleConfig()
    findings = {k: storage.read(FINDINGS, k) for k in storage.keys(FINDINGS)}
    profiles = {k: storage.read(PROFILES, k) for k in findings}
    annotations = {k: storage.read(ANNOTATIONS, k) for k in storage.keys(ANNOTATIONS)}
    links = (storage.read(LINKS, "links") or {"links": []})["links"]
    term_store = storage.read(TERMS, "terms")
    if not term_store:
        raise RuntimeError("No terms stored; run the annotate stage first")

    proposed = [t for t in term_store["terms"] if t["status"] == "proposed"]
    derived = derive_linked_terms(proposed, findings, links, adapter, cfg)
    overrides = load_overrides(cfg.overrides_path)
    candidates, applied = apply_term_overrides(proposed + derived, overrides)

    # overridden and derived terms get the same name and value checks as learned ones
    by_table_links: dict[str, set[str]] = {}
    for l in links:
        if l["status"] == "accepted" and l["best"]:
            by_table_links.setdefault(l["child"]["table"], set()).add(l["parent"]["table"])
    checked = []
    for t in candidates:
        allowed = {k: profiles[k] for k in {t["table"]} | by_table_links.get(t["table"], set())}
        problems = check_sql(t["sql"], t["kind"], t["table"], allowed, adapter.sqlglot_dialect)
        checked.append((t, problems))

    def work(item):
        term, problems = item
        if problems:
            return {"id": term["id"], "status": "unverified", "reason": "; ".join(problems)}
        return verify_term(adapter, term, findings, profiles, cfg)

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        results = list(pool.map(work, checked))
    storage.write(VERIFICATIONS, "verifications", {"assemble_version": cfg.version, "results": results})

    verified, held_back = [], []
    for (term, _), result in zip(checked, results):
        entry = {**{k: v for k, v in term.items() if k not in ("status", "issues")}, "verification": result}
        (verified if result["status"] == "verified" else held_back).append(entry)

    tables = {
        k: _table_section(k, findings[k], profiles[k], (annotations.get(k) or {}).get("annotation"), cfg)
        for k in sorted(findings)
    }
    applied += _apply_column_overrides(tables, overrides)
    verified_ids = {t["id"] for t in verified}
    groups = []
    for g in term_store["groups"]:
        ids = [i for i in g["term_ids"] if i in verified_ids]
        if ids:
            groups.append({**g, "term_ids": ids})

    annotation_meta = sorted({(a["model"], a["prompt_version"]) for a in annotations.values()})
    layer = {
        "sources": {
            "detector_version": next(iter(findings.values()))["detector_version"],
            "annotate": [{"model": m, "prompt_version": p} for m, p in annotation_meta],
            "assemble_version": cfg.version,
        },
        "tables": tables,
        "links": [
            {k: l[k] for k in ("child", "parent", "containment", "parent_coverage", "cardinality", "notes", "best", "ambiguous")}
            for l in links
            if l["status"] == "accepted"
        ],
        "terms": sorted(verified, key=lambda t: t["id"]),
        "held_back_terms": sorted(held_back, key=lambda t: t["id"]),
        "groups": groups,
        "overrides": applied,
    }
    text = json.dumps(layer, sort_keys=True, ensure_ascii=False)
    version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    current = storage.read(LAYERS, "current")
    history = (storage.read(LAYERS, "history") or {"versions": []})["versions"]
    unchanged = bool(current) and current["version"] == version
    if not unchanged:
        previous = storage.read(LAYERS, current["version"]) if current else None
        old = {t["id"]: t for t in (previous or {}).get("terms", [])}
        new = {t["id"]: t for t in layer["terms"]}
        published_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        history.append(
            {
                "version": version,
                "previous": current["version"] if current else None,
                "published_at": published_at,
                "verified": len(verified),
                "unverified": len(held_back),
                "added": sorted(set(new) - set(old)),
                "removed": sorted(set(old) - set(new)),
                "changed": sorted(i for i in set(new) & set(old) if new[i]["sql"] != old[i]["sql"]),
                "overrides": [{k: o[k] for k in ("action", "id", "reason") if k in o} | ({"error": o["error"]} if o.get("error") else {}) for o in applied],
            }
        )
        # the version file first, then the pointer: a reader sees the old layer or the new, never half
        storage.write(LAYERS, version, {"version": version, **layer})
        storage.write(LAYERS, "history", {"versions": history})
        storage.write(LAYERS, "current", {"version": version, "published_at": published_at})

        out = Path(cfg.export_dir)
        out.mkdir(parents=True, exist_ok=True)
        export = yaml.safe_dump({"version": version, **layer}, sort_keys=True, allow_unicode=True, width=110)
        (out / "semantic_layer.yaml").write_text(export, encoding="utf-8", newline="\n")
        (out / "CHANGELOG.md").write_text(_changelog(history), encoding="utf-8", newline="\n")

    reasons: dict[str, int] = {}
    for t in held_back:
        reason = re.sub(r"\d+", "N", t["verification"]["reason"])[:80]
        reasons[reason] = reasons.get(reason, 0) + 1
    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "stage": "assemble",
        "config": asdict(cfg),
        "version": version,
        "unchanged": unchanged,
        "terms": {"proposed": len(proposed), "derived": len(derived), "verified": len(verified), "held_back": len(held_back)},
        "held_back_reasons": dict(sorted(reasons.items(), key=lambda r: -r[1])),
        "overrides_applied": len([o for o in applied if not o.get("error")]),
    }
    storage.write(RUNS, f"assemble-{run['run_id']}", run)
    return run
