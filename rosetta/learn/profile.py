"""Profile: measure the data.

Reads the catalog and scans each table through the adapter. Records per column:
empty count, distinct count, min, max and a value list (every value when there
are few, otherwise the most frequent ones). Nothing here uses an LLM, and the
output carries no timestamps, so unchanged data gives byte-identical profiles.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from rosetta.adapters.base import Column, SourceAdapter
from rosetta.storage import Storage

PROFILES = "profiles"
RUNS = "runs"

# Common names for identifiers and personal data across industries (contact,
# government, financial, telecom and device identifiers). Values of such
# columns are measured but never stored, so they cannot reach a prompt or the
# repository. Abbreviations match by prefix, as elsewhere (CUST / CUSTOMER).
_SENSITIVE_TOKENS = {
    "EMAIL", "PHONE", "MOBILENO", "FAX", "ADDRESS", "POSTCODE", "ZIPCODE",
    "SSN", "PASSPORT", "NATIONALID", "TAXID", "LICENSE", "BIRTH", "DOB", "SURNAME",
    "PASSWORD", "IBAN", "SWIFT", "CARDNO", "CVV",
    "MSISDN", "IMSI", "IMEI", "ICCID",
}
_PERSON_PREFIXES = {"USER", "FIRST", "LAST", "FULL", "PERSON", "CONTACT"}


def _matches(token: str, term: str) -> bool:
    return token == term or (min(len(token), len(term)) >= 4 and (token.startswith(term) or term.startswith(token)))


@dataclass(frozen=True)
class ProfileConfig:
    value_list_max: int = 200  # list every value up to this many distinct values
    top_n: int = 20  # otherwise keep this many most frequent values
    sample_threshold_rows: int = 5_000_000  # sample tables larger than this
    sample_percent: float = 1.0
    sample_seed: int = 42
    timeout_seconds: int = 120
    workers: int = 4

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def sensitive_reason(column_name: str) -> str | None:
    parts = column_name.upper().split("_")
    tokens = set(parts) | {a + b for a, b in zip(parts, parts[1:])}  # TAX_ID also reads as TAXID
    hit = sorted({term for term in _SENSITIVE_TOKENS for t in tokens if _matches(t, term)})
    if hit:
        return f"identifier or personal-data term {hit[0]}"
    if "NAME" in tokens and tokens & _PERSON_PREFIXES:
        return "person name"
    return None


def to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def fingerprint(columns: list[Column], num_rows: int | None) -> str:
    """Cheap catalog-level identity of a table: its columns and catalog row count."""
    shape = [(c.name, c.data_type, c.nullable) for c in sorted(columns, key=lambda c: c.position)]
    return hashlib.sha256(json.dumps([shape, num_rows]).encode()).hexdigest()[:16]


def profile_table(
    adapter: SourceAdapter,
    schema: str,
    table: str,
    columns: list[Column],
    num_rows: int | None,
    cfg: ProfileConfig,
) -> dict[str, Any]:
    columns = sorted(columns, key=lambda c: c.position)
    q = adapter.quote
    source = f"{q(schema)}.{q(table)}"

    if num_rows is None:
        num_rows = adapter.run_select(
            f"SELECT COUNT(*) FROM {source}", timeout=cfg.timeout_seconds
        ).rows[0][0]
    sampled = num_rows > cfg.sample_threshold_rows
    from_clause = f"{source} {adapter.sample_clause(cfg.sample_percent, cfg.sample_seed)}" if sampled else source

    name_flags = {c.name: sensitive_reason(c.name) for c in columns}

    # one scan per table for every column's counts and range
    select_items = ["COUNT(*)"]
    for c in columns:
        distinct = adapter.approx_distinct_expr(c.name) if sampled else f"COUNT(DISTINCT {q(c.name)})"
        select_items += [f"COUNT({q(c.name)})", distinct, f"MIN({q(c.name)})", f"MAX({q(c.name)})"]
    scan = adapter.run_select(
        f"SELECT {', '.join(select_items)} FROM {from_clause}", timeout=cfg.timeout_seconds
    ).rows[0]
    total = int(scan[0])

    profiled = []
    cursor = 1
    for c in columns:
        non_null, distinct = int(scan[cursor]), int(scan[cursor + 1])
        low, high = to_json_value(scan[cursor + 2]), to_json_value(scan[cursor + 3])
        cursor += 4

        # A name hint alone is not enough: flags and dates carry business meaning.
        # Mask person names always, and identifiers only when they are not dates
        # and have too many distinct values to be a code list.
        reason = name_flags[c.name]
        masked = bool(reason) and (
            reason == "person name"
            or (c.data_type != "DATE" and distinct > cfg.value_list_max)
        )
        if masked:
            low = high = None

        values, complete, note = [], False, None
        if masked:
            note = "masked: " + reason
        elif non_null == 0:
            note = "no values"
        elif distinct > cfg.value_list_max and distinct == non_null:
            note = "unique per row; no value list"
        else:
            keep = cfg.value_list_max if distinct <= cfg.value_list_max else cfg.top_n
            complete = distinct <= cfg.value_list_max
            rows = adapter.run_select(
                f"SELECT {q(c.name)}, COUNT(*) FROM {from_clause} WHERE {q(c.name)} IS NOT NULL "
                f"GROUP BY {q(c.name)} ORDER BY COUNT(*) DESC, {q(c.name)} "
                f"{adapter.limit_clause(keep)}",
                timeout=cfg.timeout_seconds,
                max_rows=keep,
            ).rows
            values = [{"value": to_json_value(v), "count": int(n)} for v, n in rows]

        profiled.append(
            {
                "name": c.name,
                "position": c.position,
                "data_type": c.data_type,
                "nullable": c.nullable,
                "comment": c.comment,
                "non_null_count": non_null,
                "empty_count": total - non_null,  # Oracle stores '' as NULL
                "distinct_count": distinct,
                "distinct_is_approximate": sampled,
                "min": low,
                "max": high,
                "sensitive": masked,
                "values": values,
                "values_complete": complete,
                "values_note": note,
            }
        )

    return {
        "schema": schema,
        "table": table,
        "fingerprint": fingerprint(columns, num_rows),
        "config_digest": cfg.digest(),
        "row_count": total,
        "sampled": sampled,
        "columns": profiled,
    }


def run_profile(
    adapter: SourceAdapter,
    storage: Storage,
    cfg: ProfileConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Profile every table. Tables whose fingerprint is unchanged are reused."""
    cfg = cfg or ProfileConfig()
    tables = adapter.list_tables()
    by_table: dict[tuple[str, str], list[Column]] = {t: [] for t in tables}
    for c in adapter.list_columns():
        by_table.setdefault((c.schema, c.table), []).append(c)
    row_counts = adapter.table_row_counts()

    status: dict[str, dict[str, Any]] = {}
    todo = []
    for schema, table in tables:
        key = f"{schema}.{table}"
        columns = by_table[(schema, table)]
        if not columns:
            status[key] = {"status": "skipped", "reason": "no columns visible"}
            continue
        existing = storage.read(PROFILES, key)
        current = fingerprint(columns, row_counts.get((schema, table)))
        if (
            not force
            and existing
            and existing["fingerprint"] == current
            and existing["config_digest"] == cfg.digest()
        ):
            status[key] = {"status": "reused"}
            continue
        todo.append((schema, table, key, columns))

    def work(item):
        schema, table, key, columns = item
        try:
            profile = profile_table(adapter, schema, table, columns, row_counts.get((schema, table)), cfg)
            storage.write(PROFILES, key, profile)
            return key, {"status": "profiled", "row_count": profile["row_count"]}
        except Exception as exc:  # recorded, never hidden
            return key, {"status": "skipped", "reason": adapter.map_error(exc)}

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for key, result in pool.map(work, todo):
            status[key] = result

    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "stage": "profile",
        "config": asdict(cfg),
        "config_digest": cfg.digest(),
        "tables": dict(sorted(status.items())),
    }
    storage.write(RUNS, f"profile-{run['run_id']}", run)
    return run
