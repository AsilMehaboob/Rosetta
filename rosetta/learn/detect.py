"""Detect: find structure.

Reads the stored profiles and, with a few confirming queries through the
adapter, works out for every table what one row stands for, whether it keeps
history or repeated snapshots, which row is current, and how tables link.

Every rule is about the shape of the data (uniqueness, date order, a far-future
end date, value overlap), never about what a table or column is called, so it
works on any database. Names only break ties between candidates the data
already supports. No LLM, and the same profiles always give the same findings.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any

from rosetta.adapters.base import SourceAdapter
from rosetta.learn.profile import PROFILES, RUNS, to_json_value
from rosetta.storage import Storage

FINDINGS = "findings"
LINKS = "links"


@dataclass(frozen=True)
class DetectConfig:
    version: str = "detect-3"
    code_max_distinct: int = 50  # a column with this few values is a code list
    lookup_max_rows: int = 50  # tables this small are reference lists; their rows are kept
    entity_options: int = 12  # entity columns (or column pairs) tried per table
    sentinel_gap_years: int = 50  # a max date this far past the next one means "never ends"
    max_overlap_share: float = 0.005  # history versions of one entity may overlap on this share of rows
    snapshot_max_dates: int = 400
    snapshot_min_rows_per_date: int = 10
    snapshot_min_coverage: float = 0.5  # share of entity x date slots that hold a row
    versioned_max_dates: int = 4
    link_min_name_score: float = 0.5
    link_min_containment: float = 0.9  # share of child values found in the parent
    value_only_min_distinct: int = 100  # links without a name match need large value sets
    value_only_min_containment: float = 0.98
    value_only_max_parents: int = 3
    ambiguity_margin: float = 0.1
    # generic naming conventions, used only to break ties
    key_hint_tokens: tuple[str, ...] = ("ID", "KEY", "CODE", "CD", "NO", "NUM", "NBR")
    timeout_seconds: int = 120
    workers: int = 4

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# names: tie-breakers only

def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^A-Z0-9]+", name.upper()) if len(t) > 1}


def _token_match(a: str, b: str) -> bool:
    # equal, or one is an abbreviation of the other (SUBSCR / SUBSCRIPTION)
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def name_score(child_column: str, parent_table: str, parent_column: str, cfg: DetectConfig) -> float:
    """How much a child column's name points at a parent column, from 0 to 1."""
    if child_column.upper() == parent_column.upper():
        return 1.0
    parent = _tokens(parent_column)
    if parent <= set(cfg.key_hint_tokens):  # a bare "ID" takes its meaning from the table
        parent |= _tokens(parent_table)
    child = _tokens(child_column)
    if not parent or not child:
        return 0.0
    hit_parent = sum(1 for p in parent if any(_token_match(p, c) for c in child))
    hit_child = sum(1 for c in child if any(_token_match(p, c) for p in parent))
    cover = hit_parent / len(parent)
    overlap = hit_parent / (len(parent) + len(child) - hit_child)
    return round(0.5 * cover + 0.5 * overlap, 3)


def _entity_hint(column: str, table: str, cfg: DetectConfig) -> float:
    column_tokens = _tokens(column)
    table_tokens = _tokens(table) - set(cfg.key_hint_tokens)
    score = 0.0
    if column_tokens & set(cfg.key_hint_tokens):
        score += 1
    if any(_token_match(c, t) for c in column_tokens for t in table_tokens):
        score += 0.75
    return score


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class _Table:
    """One profiled table plus the query helpers Detect needs."""

    def __init__(self, adapter: SourceAdapter, profile: dict[str, Any], cfg: DetectConfig):
        self.adapter, self.cfg, self.profile = adapter, cfg, profile
        self.schema, self.name = profile["schema"], profile["table"]
        self.key = f"{self.schema}.{self.name}"
        self.rows = profile["row_count"]
        self.exact = not profile["sampled"]
        self.src = f"{adapter.quote(self.schema)}.{adapter.quote(self.name)}"
        self.cols = {c["name"]: c for c in profile["columns"]}
        self.family = {n: adapter.type_family(c["data_type"]) for n, c in self.cols.items()}

    def q(self, column: str) -> str:
        return self.adapter.quote(column)

    def full(self, column: str) -> bool:
        return self.rows > 0 and self.cols[column]["non_null_count"] == self.rows

    def distinct(self, column: str) -> int:
        return self.cols[column]["distinct_count"]

    def select(self, sql: str, max_rows: int = 1) -> tuple:
        return self.adapter.run_select(sql, timeout=self.cfg.timeout_seconds, max_rows=max_rows).rows

    def scalar_row(self, sql: str) -> tuple:
        return self.select(sql)[0]

    def dup_groups(self, columns: tuple[str, ...], where: str = "", extra_from: str = "") -> int:
        """How many value combinations of these columns appear on more than one row."""
        group = ", ".join(self.q(c) for c in columns)
        where_sql = f" WHERE {where}" if where else ""
        sql = (
            f"SELECT COUNT(*) FROM (SELECT {group} FROM {self.src}{extra_from}{where_sql} "
            f"GROUP BY {group} HAVING COUNT(*) > 1) d"
        )
        return int(self.scalar_row(sql)[0])

    def group_count(self, columns: tuple[str, ...]) -> int:
        if len(columns) == 1 and self.exact:
            return self.distinct(columns[0])
        group = ", ".join(self.q(c) for c in columns)
        return int(self.scalar_row(f"SELECT COUNT(*) FROM (SELECT {group} FROM {self.src} GROUP BY {group}) d")[0])


def _shape(t: _Table, column: str, cfg: DetectConfig) -> dict[str, Any]:
    c = t.cols[column]
    filled, distinct = c["non_null_count"], c["distinct_count"]
    if filled == 0:
        shape = "empty"
    elif distinct == 1:
        shape = "constant"
    elif t.family[column] == "date":
        shape = "date"
    elif distinct == 2:
        shape = "flag"
    elif distinct <= cfg.code_max_distinct:
        shape = "code"
    elif distinct == filled == t.rows:
        shape = "unique"
    else:
        shape = t.family[column]
    return {"shape": shape, "filled_share": round(filled / t.rows, 4) if t.rows else 0.0}


def _entity_options(t: _Table, cfg: DetectConfig) -> list[tuple[str, ...]]:
    """Columns (or a column plus a code) that could name the thing a table tracks."""
    singles = [
        n
        for n in t.cols
        if t.family[n] in ("number", "text")
        and t.full(n)
        and 2 <= t.distinct(n) < t.rows
        and (t.distinct(n) > cfg.code_max_distinct or 2 * t.distinct(n) > t.rows)
    ]
    singles.sort(key=lambda n: (-_entity_hint(n, t.name, cfg), t.cols[n]["sensitive"], -t.distinct(n), n))
    codes = sorted(
        n
        for n in t.cols
        if t.family[n] in ("number", "text") and t.full(n) and 2 <= t.distinct(n) <= cfg.code_max_distinct
    )
    half = cfg.entity_options // 2
    pairs = [(e, c) for e in singles[:2] for c in codes if c != e]
    return [(n,) for n in singles[:half]] + pairs[: cfg.entity_options - min(half, len(singles))]


def _find_snapshot(t: _Table, cfg: DetectConfig) -> dict[str, Any] | None:
    """Repeated copies: the same entities written again for each of a few dates."""
    dates = sorted(
        (
            n
            for n in t.cols
            if t.family[n] == "date"
            and t.full(n)
            and 2 <= t.distinct(n) <= cfg.snapshot_max_dates
            and t.rows / t.distinct(n) >= cfg.snapshot_min_rows_per_date
        ),
        key=lambda n: (t.distinct(n), n),
    )
    for d in dates:
        n_dates = t.distinct(d)
        found = []
        for rank, entity in enumerate(_entity_options(t, cfg)):
            if len(entity) == 1:  # cheap check before any query
                slots = t.distinct(entity[0]) * n_dates
                if slots < t.rows or t.rows / slots < cfg.snapshot_min_coverage:
                    continue
            if t.dup_groups(entity + (d,)):
                continue
            coverage = t.rows / (t.group_count(entity) * n_dates)
            if coverage >= cfg.snapshot_min_coverage:
                found.append((-round(coverage, 3), rank, entity, coverage))
        if not found:
            continue
        _, _, entity, coverage = min(found)
        per_date = t.select(
            f"SELECT {t.q(d)}, COUNT(*) FROM {t.src} GROUP BY {t.q(d)} ORDER BY {t.q(d)}",
            max_rows=cfg.snapshot_max_dates,
        )
        dates_seen = [{"date": to_json_value(v), "rows": int(n)} for v, n in per_date]
        latest = dates_seen[-1]["date"]
        return {
            "entity": list(entity),
            "date_column": d,
            "dates": dates_seen,
            "coverage": round(coverage, 4),
            "evidence": f"each {'+'.join(entity)} appears at most once per {d}; "
            f"{len(dates_seen)} dates, {round(coverage * 100, 1)}% of entity x date slots filled",
            "current_rule": {"type": "latest_snapshot", "column": d, "value": latest},
            "current_sql": f"{t.q(d)} = (SELECT MAX({t.q(d)}) FROM {t.src})",
            "rule_text": f"each {'+'.join(entity)} appears once per {d}; for the current state use "
            f"{d} = {latest}; counting entities across several {d} values counts each one "
            "several times",
        }
    return None


def _sentinel(t: _Table, column: str, cfg: DetectConfig) -> dict[str, Any] | None:
    """A shared far-future date standing for "never ends" (4712-12-31, 9999-12-31, ...)."""
    x = t.q(column)
    top, next_latest, at_top = t.scalar_row(
        f"SELECT MAX(m.mx), MAX(CASE WHEN {x} < m.mx THEN {x} END), "
        f"SUM(CASE WHEN {x} = m.mx THEN 1 ELSE 0 END) "
        f"FROM {t.src}, (SELECT MAX({x}) AS mx FROM {t.src}) m"
    )
    top_dt, next_dt = _as_datetime(top), _as_datetime(next_latest)
    if top_dt is None or next_dt is None or not at_top or int(at_top) < 2:
        return None
    gap_years = (top_dt - next_dt).days / 365.25
    if gap_years < cfg.sentinel_gap_years:
        return None
    return {"value": to_json_value(top), "rows": int(at_top), "next_latest": to_json_value(next_latest)}


def _open_sql(t: _Table, end: str, sentinel: dict | None, null_open: bool) -> tuple[str, str]:
    """(extra FROM, predicate) that marks rows whose end date says "still current"."""
    parts, extra = [], ""
    if sentinel:
        extra = f", (SELECT MAX({t.q(end)}) AS mx FROM {t.src}) m"
        parts.append(f"{t.q(end)} = m.mx")
    if null_open:
        parts.append(f"{t.q(end)} IS NULL")
    return extra, "(" + " OR ".join(parts) + ")"


def _find_history(t: _Table, cfg: DetectConfig) -> dict[str, Any] | None:
    """Versions of an entity with a start and an end date, at most one of them open."""
    date_cols = sorted(n for n in t.cols if t.family[n] == "date" and t.cols[n]["non_null_count"] > 0)
    starts = [n for n in date_cols if t.full(n) and t.distinct(n) >= 2]
    ends = []
    for x in date_cols:
        sentinel = _sentinel(t, x, cfg) if t.cols[x]["max"] is not None else None
        null_open = t.cols[x]["non_null_count"] < t.rows
        if sentinel or null_open:
            ends.append((x, sentinel, null_open))
    pairs = []
    for s in starts:
        for x, sentinel, null_open in ends:
            s_min, x_max = _as_datetime(t.cols[s]["min"]), _as_datetime(t.cols[x]["max"])
            if s == x or (s_min and x_max and s_min > x_max):
                continue
            similarity = len(_tokens(s) & _tokens(x))
            pairs.append((-similarity, 0 if sentinel else 1, s, x, sentinel, null_open))
    pairs.sort(key=lambda p: p[:4])

    options = _entity_options(t, cfg)
    for _, _, s, x, sentinel, null_open in pairs:
        extra, open_pred = _open_sql(t, x, sentinel, null_open)
        bad_order, open_rows = t.scalar_row(
            f"SELECT SUM(CASE WHEN {t.q(x)} IS NOT NULL AND {t.q(s)} > {t.q(x)} THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN {open_pred} THEN 1 ELSE 0 END) FROM {t.src}{extra}"
        )
        open_rows = int(open_rows or 0)
        if int(bad_order or 0) or not 0 < open_rows < t.rows:
            continue
        for entity in options:
            if t.dup_groups(entity, where=open_pred, extra_from=extra):
                continue  # some entity has two open versions
            partition = ", ".join(t.q(c) for c in entity)
            successors, overlaps, touching = t.scalar_row(
                "SELECT COUNT(*), SUM(CASE WHEN r_end IS NULL OR r_next < r_end THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN r_next = r_end THEN 1 ELSE 0 END) FROM ("
                f"SELECT {t.q(x)} AS r_end, LEAD({t.q(s)}) OVER (PARTITION BY {partition} "
                f"ORDER BY {t.q(s)}, {t.q(x)}) AS r_next FROM {t.src}) w WHERE r_next IS NOT NULL"
            )
            successors, overlaps, touching = int(successors), int(overlaps or 0), int(touching or 0)
            if successors == 0 or overlaps > cfg.max_overlap_share * t.rows:
                continue
            touching_end = touching * 2 >= successors
            open_desc = " or ".join(
                ([f"{x} = {sentinel['value']}"] if sentinel else []) + ([f"{x} is empty"] if null_open else [])
            )
            end_cmp = "<" if touching_end else "<="
            # ready-made SQL, so later steps copy the exact marker (time of day included)
            open_parts = ([f"{t.q(x)} = {t.adapter.datetime_literal(_as_datetime(sentinel['value']))}"]
                          if sentinel else []) + ([f"{t.q(x)} IS NULL"] if null_open else [])
            current_sql = open_parts[0] if len(open_parts) == 1 else "(" + " OR ".join(open_parts) + ")"
            as_of_sql = (
                f"{t.q(s)} <= {{as_of}} AND ({' OR '.join(open_parts)} OR {{as_of}} {end_cmp} {t.q(x)})"
            )
            return {
                "entity": list(entity),
                "start": s,
                "end": x,
                "open_marker": {"end_value": sentinel["value"] if sentinel else None, "end_is_null": null_open},
                "evidence": {
                    "rows": t.rows,
                    "open_rows": open_rows,
                    "rows_with_a_later_version": successors,
                    "overlapping_versions": overlaps,
                    "end_equals_next_start": touching,
                    "start_after_end_rows": 0,
                    "sentinel": sentinel,
                },
                "boundary": "end_equals_next_start" if touching_end else "end_before_next_start",
                "current_rule": {
                    "type": "open_end",
                    "column": x,
                    "open_values": [sentinel["value"]] if sentinel else [],
                    "null_is_open": null_open,
                },
                "as_of_rule": {"type": "interval", "start": s, "end": x, "end_inclusive": not touching_end},
                "current_sql": current_sql,
                "as_of_sql": as_of_sql,
                "rule_text": f"one version per {'+'.join(entity)} at a time; current version: {open_desc}; "
                f"as of date D: {s} <= D and ({open_desc} or D {end_cmp} {x})",
            }
    return None


def _find_versioned(t: _Table, cfg: DetectConfig) -> dict[str, Any] | None:
    """No single unique column, but entity + one date is unique."""
    dates = sorted(
        (n for n in t.cols if t.family[n] == "date" and t.full(n)), key=lambda n: (-t.distinct(n), n)
    )[: cfg.versioned_max_dates]
    for entity in _entity_options(t, cfg)[: cfg.entity_options // 2]:
        for d in dates:
            if not t.dup_groups(entity + (d,)):
                return {
                    "entity": list(entity),
                    "date_column": d,
                    "latest_rule": {"type": "latest_per_entity", "entity": list(entity), "column": d},
                    "rule_text": f"several rows per {'+'.join(entity)} (versions or events), one per {d}; "
                    f"the latest {d} per {'+'.join(entity)} is the most recent row",
                }
    return None


def _code_behaviour(t: _Table, cfg: DetectConfig, extra: str, current_pred: str, skip: set[str]) -> dict[str, Any]:
    """For each code column: rows per value, and how many of them are current."""
    out = {}
    for c in sorted(t.cols):
        col = t.cols[c]
        if (
            c in skip
            or col["sensitive"]
            or t.family[c] not in ("number", "text")
            or not 2 <= t.distinct(c) <= cfg.code_max_distinct
        ):
            continue
        rows = t.select(
            f"SELECT {t.q(c)}, COUNT(*), SUM(CASE WHEN {current_pred} THEN 1 ELSE 0 END) "
            f"FROM {t.src}{extra} GROUP BY {t.q(c)}",
            max_rows=cfg.code_max_distinct + 1,
        )
        values = [
            {"value": to_json_value(v), "rows": int(n), "current_rows": int(k or 0)} for v, n, k in rows
        ]
        values.sort(key=lambda r: (r["value"] is None, str(r["value"])))
        out[c] = values
    return out


def detect_table(adapter: SourceAdapter, profile: dict[str, Any], cfg: DetectConfig) -> dict[str, Any]:
    t = _Table(adapter, profile, cfg)
    finding: dict[str, Any] = {
        "schema": t.schema,
        "table": t.name,
        "row_count": t.rows,
        "profile_fingerprint": profile["fingerprint"],
        "detector_version": cfg.version,
        "config_digest": cfg.digest(),
        "columns": {n: _shape(t, n, cfg) for n in sorted(t.cols)},
        "unique_columns": [],
        "kind": "empty",
        "grain": None,
        "snapshot": None,
        "history": None,
        "versioned": None,
        "code_behaviour": {},
        "lookup_rows": None,
    }
    if t.rows == 0:
        return finding

    if t.rows <= cfg.lookup_max_rows:
        # a small reference list (statuses, types) means little without its rows;
        # personal-data columns are left out
        shown = sorted(n for n in t.cols if not t.cols[n]["sensitive"])
        order = ", ".join(t.q(n) for n in shown)
        rows = t.select(f"SELECT {order} FROM {t.src} ORDER BY {order}", max_rows=cfg.lookup_max_rows)
        finding["lookup_rows"] = {
            "columns": shown,
            "rows": [[to_json_value(v) for v in r] for r in rows],
        }

    unique = [n for n in t.cols if t.full(n) and t.distinct(n) == t.rows]
    if not t.exact:
        unique = [n for n in unique if not t.dup_groups((n,))]
    unique.sort(
        key=lambda n: (t.family[n] == "date", -_entity_hint(n, t.name, cfg), t.cols[n]["sensitive"], n)
    )
    finding["unique_columns"] = unique

    snapshot = _find_snapshot(t, cfg)
    history = None if snapshot else _find_history(t, cfg)
    unique_non_date = [n for n in unique if t.family[n] != "date"]
    versioned = None if (snapshot or history or unique_non_date) else _find_versioned(t, cfg)

    if snapshot:
        entity, d = tuple(snapshot["entity"]), snapshot["date_column"]
        finding.update(kind="snapshot", snapshot=snapshot)
        finding["grain"] = {
            "columns": list(entity) + [d],
            "verified": True,
            "text": f"one row per {'+'.join(entity)} per {d}",
        }
        finding["code_behaviour"] = _code_behaviour(
            t, cfg, f", (SELECT MAX({t.q(d)}) AS mx FROM {t.src}) m", f"{t.q(d)} = m.mx", set(entity)
        )
    elif history:
        entity = tuple(history["entity"])
        finding.update(kind="history", history=history)
        verified = not t.dup_groups(entity + (history["start"],))
        finding["grain"] = {
            "columns": list(entity) + [history["start"]],
            "verified": verified,
            "text": f"one row per version of {'+'.join(entity)}, starting at {history['start']}",
        }
        marker = history["open_marker"]
        sentinel = {"value": marker["end_value"]} if marker["end_value"] is not None else None
        extra, open_pred = _open_sql(t, history["end"], sentinel, marker["end_is_null"])
        finding["code_behaviour"] = _code_behaviour(t, cfg, extra, open_pred, set(entity))
    elif unique_non_date:
        finding["kind"] = "keyed"
        finding["grain"] = {
            "columns": [unique_non_date[0]],
            "verified": True,
            "text": f"one row per {unique_non_date[0]}",
        }
    elif versioned:
        finding.update(kind="entity_by_date", versioned=versioned)
        finding["grain"] = {
            "columns": versioned["entity"] + [versioned["date_column"]],
            "verified": True,
            "text": f"one row per {'+'.join(versioned['entity'])} per {versioned['date_column']}",
        }
    elif unique:
        finding["kind"] = "keyed"
        finding["grain"] = {"columns": [unique[0]], "verified": True, "text": f"one row per {unique[0]}"}
    else:
        finding["kind"] = "unkeyed"
        finding["grain"] = {
            "columns": [],
            "verified": False,
            "text": "no unique column and no entity + date combination found; count with care",
        }
    return finding


# links between tables

def _key_columns(finding: dict[str, Any], profile: dict[str, Any], adapter: SourceAdapter) -> dict[str, float]:
    """Columns another table could point at, with how many rows each key value has."""
    cols = {c["name"]: c for c in profile["columns"]}
    family = {n: adapter.type_family(c["data_type"]) for n, c in cols.items()}
    rows = finding["row_count"]
    keys = {n: 1.0 for n in finding["unique_columns"] if family[n] in ("number", "text")}
    detail = finding["snapshot"] or finding["history"] or finding["versioned"]
    if detail:
        entity = detail["entity"][0]
        same = cols[entity]["distinct_count"]
        # columns with exactly as many values as the entity are likely alternate names for it
        for n, c in cols.items():
            if (
                family[n] == family[entity]
                and c["non_null_count"] == rows
                and c["distinct_count"] == same
                and n not in keys
            ):
                keys[n] = round(rows / same, 2)
    return keys


def _containment(adapter: SourceAdapter, child: dict, parent: dict, cfg: DetectConfig) -> tuple[int, int]:
    q = adapter.quote
    c_src = f"{q(child['schema'])}.{q(child['table'])}"
    p_src = f"{q(parent['schema'])}.{q(parent['table'])}"
    c_col, p_col = q(child["column"]), q(parent["column"])
    row = adapter.run_select(
        f"SELECT COUNT(*), COUNT(p.v) FROM (SELECT DISTINCT {c_col} AS v FROM {c_src} "
        f"WHERE {c_col} IS NOT NULL) c LEFT JOIN (SELECT DISTINCT {p_col} AS v FROM {p_src}) p ON c.v = p.v",
        timeout=cfg.timeout_seconds,
        max_rows=1,
    ).rows[0]
    return int(row[0]), int(row[1])


def detect_links(
    adapter: SourceAdapter,
    profiles: dict[str, dict[str, Any]],
    findings: dict[str, dict[str, Any]],
    cfg: DetectConfig,
) -> list[dict[str, Any]]:
    columns = {}
    for key, profile in profiles.items():
        for c in profile["columns"]:
            columns[(key, c["name"])] = {
                "schema": profile["schema"],
                "table": profile["table"],
                "table_key": key,
                "column": c["name"],
                "family": adapter.type_family(c["data_type"]),
                "distinct": c["distinct_count"],
                "min": c["min"],
                "max": c["max"],
                "unique": c["name"] in findings[key]["unique_columns"],
            }
    parents = []
    for key, finding in findings.items():
        for name, rows_per_key in _key_columns(finding, profiles[key], adapter).items():
            parents.append({**columns[(key, name)], "rows_per_key": rows_per_key, "kind": finding["kind"]})
    children = [
        c for c in columns.values() if c["family"] in ("number", "text") and c["distinct"] >= 3
    ]

    def in_range(child: dict, parent: dict, inside: bool) -> bool:
        if child["family"] != "number" or None in (child["min"], child["max"], parent["min"], parent["max"]):
            return True
        if inside:
            return parent["min"] <= child["min"] and child["max"] <= parent["max"]
        return child["min"] <= parent["max"] and parent["min"] <= child["max"]

    candidates = []
    for child in sorted(children, key=lambda c: (c["table_key"], c["column"])):
        named, value_only = [], []
        for parent in parents:
            if parent["table_key"] == child["table_key"] or parent["family"] != child["family"]:
                continue
            if child["distinct"] * cfg.link_min_containment > parent["distinct"]:
                continue  # too many child values to fit inside the parent
            score = name_score(child["column"], parent["table"], parent["column"], cfg)
            if score >= cfg.link_min_name_score and in_range(child, parent, inside=False):
                named.append((child, parent, score, "named"))
            elif (
                min(child["distinct"], parent["distinct"]) >= cfg.value_only_min_distinct
                and in_range(child, parent, inside=True)
            ):
                value_only.append((child, parent, score, "values"))
        if named:
            candidates += named
        else:
            value_only.sort(key=lambda c: (abs(c[1]["distinct"] - child["distinct"]), c[1]["table_key"], c[1]["column"]))
            candidates += value_only[: cfg.value_only_max_parents]

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        counts = list(pool.map(lambda c: _containment(adapter, c[0], c[1], cfg), candidates))

    links = []
    for (child, parent, score, match), (child_values, matched) in zip(candidates, counts):
        containment = matched / child_values if child_values else 0.0
        needed = cfg.link_min_containment if match == "named" else cfg.value_only_min_containment
        links.append(
            {
                "child": {"table": child["table_key"], "column": child["column"]},
                "parent": {"table": parent["table_key"], "column": parent["column"]},
                "match": match,
                "name_score": score,
                "child_values": child_values,
                "matched_values": matched,
                "containment": round(containment, 4),
                "parent_values": parent["distinct"],
                "parent_coverage": round(matched / parent["distinct"], 4) if parent["distinct"] else 0.0,
                "parent_kind": parent["kind"],
                "parent_rows_per_key": parent["rows_per_key"],
                "cardinality": "one_to_one" if child["unique"] else "many_to_one",
                "passed": containment >= needed and matched > 0,
            }
        )

    # two key columns holding the same set of values are one shared key, not a choice
    same_domain = set()
    by_pair = {(l["child"]["table"], l["child"]["column"], l["parent"]["table"], l["parent"]["column"]): l for l in links}
    for (ct, cc, pt, pc), l in by_pair.items():
        back = by_pair.get((pt, pc, ct, cc))
        if l["containment"] >= 0.99 and back and back["containment"] >= 0.99:
            same_domain.add(((ct, cc), (pt, pc)))

    by_child: dict[tuple[str, str], list[dict]] = {}
    for l in links:
        by_child.setdefault((l["child"]["table"], l["child"]["column"]), []).append(l)
    for group in by_child.values():
        passing = sorted(
            (l for l in group if l["passed"]),
            key=lambda l: (
                l["match"] != "named",
                -l["containment"],
                -l["parent_coverage"],
                l["parent_rows_per_key"] > 1,  # a parent with one row per key cannot double rows
                -l["name_score"],
                l["parent"]["table"],
                l["parent"]["column"],
            ),
        )
        for i, l in enumerate(passing):
            l["status"] = "accepted" if l["match"] == "named" else "candidate"
            l["best"] = i == 0
            l["ambiguous"] = False
        if len(passing) > 1:
            best = passing[0]
            best_key = (best["parent"]["table"], best["parent"]["column"])
            for other in passing[1:]:
                other_key = (other["parent"]["table"], other["parent"]["column"])
                close = (
                    best["containment"] - other["containment"] < cfg.ambiguity_margin
                    and best["parent_coverage"] - other["parent_coverage"] < cfg.ambiguity_margin
                )
                if close and (best_key, other_key) not in same_domain and (other_key, best_key) not in same_domain:
                    best["ambiguous"] = other["ambiguous"] = True
        for l in group:
            if not l["passed"]:
                l["status"] = "rejected"
                l["reason"] = f"only {round(l['containment'] * 100, 1)}% of values found in the parent"
        for l in group:
            notes = []
            if l["status"] != "rejected" and l["parent_rows_per_key"] > 1:
                notes.append(
                    f"parent is a {l['parent_kind']} table with several rows per key; "
                    "restrict it to one row per key (current or as of a date) before joining, or rows double"
                )
            if l.get("ambiguous"):
                notes.append("another parent fits about as well; prefer the best one and say so")
            l["notes"] = notes

    # keep rejected value-only guesses out: they are noise, not evidence
    kept = [l for l in links if l["match"] == "named" or l["status"] != "rejected"]
    for l in kept:
        l.pop("passed")
    kept.sort(key=lambda l: (l["child"]["table"], l["child"]["column"], l["parent"]["table"], l["parent"]["column"]))
    return kept


def run_detect(adapter: SourceAdapter, storage: Storage, cfg: DetectConfig | None = None) -> dict[str, Any]:
    """Detect structure for every profiled table, then the links between them."""
    cfg = cfg or DetectConfig()
    profiles = {key: storage.read(PROFILES, key) for key in storage.keys(PROFILES)}
    if not profiles:
        raise RuntimeError("No profiles stored; run the profile stage first")

    def work(key: str):
        try:
            return key, detect_table(adapter, profiles[key], cfg), None
        except Exception as exc:  # recorded, never hidden
            return key, None, adapter.map_error(exc)

    findings, status = {}, {}
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for key, finding, error in pool.map(work, sorted(profiles)):
            if error:
                status[key] = {"status": "failed", "reason": error}
                continue
            findings[key] = finding
            storage.write(FINDINGS, key, finding)
            status[key] = {"status": "detected", "kind": finding["kind"], "grain": finding["grain"]["text"] if finding["grain"] else None}

    links = detect_links(adapter, {k: profiles[k] for k in findings}, findings, cfg)
    storage.write(
        LINKS,
        "links",
        {"detector_version": cfg.version, "config_digest": cfg.digest(), "links": links},
    )

    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "stage": "detect",
        "config": asdict(cfg),
        "config_digest": cfg.digest(),
        "tables": dict(sorted(status.items())),
        "links": {
            s: sum(1 for l in links if l["status"] == s) for s in ("accepted", "candidate", "rejected")
        },
        "ambiguous_links": sum(1 for l in links if l.get("ambiguous")),
    }
    storage.write(RUNS, f"detect-{run['run_id']}", run)
    return run
