"""Detect must find structure in a database it knows nothing about.

The source database here is a small made-up hospital in SQLite, with different
names, a different "never ends" date and a different engine from the one
Rosetta was built against. If Detect only worked because of hard-coded names,
these tests would fail.
"""

import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from rosetta.adapters.base import Column, QueryResult, SourceAdapter
from rosetta.learn.detect import FINDINGS, LINKS, DetectConfig, name_score, run_detect
from rosetta.learn.profile import ProfileConfig, run_profile
from rosetta.storage import Storage

SCHEMA = "MAIN"


class SqliteAdapter(SourceAdapter):
    sqlglot_dialect = "sqlite"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def connect_pool(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_select(self, sql, binds=None, timeout=None, max_rows=None) -> QueryResult:
        cur = self.conn.execute(sql, binds or [])
        max_rows = max_rows or 100_000
        rows = cur.fetchmany(max_rows + 1)
        columns = tuple(d[0] for d in cur.description)
        return QueryResult(columns, tuple(tuple(r) for r in rows[:max_rows]), len(rows) > max_rows)

    def list_tables(self):
        names = self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        return [(SCHEMA, n) for (n,) in names]

    def list_columns(self):
        columns = []
        for _, table in self.list_tables():
            for cid, name, data_type, notnull, _, _ in self.conn.execute(f'PRAGMA table_info("{table}")'):
                columns.append(Column(SCHEMA, table, name, data_type, not notnull, cid + 1))
        return columns

    def list_keys(self):
        return []

    def table_row_counts(self):
        return {t: None for t in self.list_tables()}

    def quote(self, identifier):
        return '"' + identifier.replace('"', '""') + '"'

    def limit_clause(self, n):
        return f"LIMIT {int(n)}"

    def date_literal(self, value):
        return f"'{value.isoformat()}'"

    def type_family(self, data_type):
        return {"DATE": "date", "INTEGER": "number", "REAL": "number", "TEXT": "text"}.get(
            data_type.upper(), "other"
        )

    def approx_distinct_expr(self, column):
        return f"COUNT(DISTINCT {self.quote(column)})"

    def sample_clause(self, percent, seed):
        return ""

    def explain_cost(self, sql):
        return None

    def map_error(self, exc):
        return f"error: {exc}"

    def dialect_rules(self):
        return ""


def _hospital() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE PATIENT (ID INTEGER, FULL_NAME TEXT, BLOOD_GROUP TEXT);
        CREATE TABLE WARD_STAY (STAY_REF TEXT, PATIENT_ID INTEGER, WARD TEXT, VALID_FROM DATE, VALID_TO DATE);
        CREATE TABLE DAILY_CENSUS (CENSUS_DAY DATE, PATIENT_ID INTEGER, CONDITION TEXT);
        """
    )
    wards = ["SURGERY", "RECOVERY", "ICU", "GENERAL"]
    for i in range(60):
        pid = 1001 + i
        conn.execute("INSERT INTO PATIENT VALUES (?, ?, ?)", (pid, f"Patient {i}", "A B AB O".split()[i % 4]))
        start = date(2024, 1, 1) + timedelta(days=i)
        stays = 1 + i % 3
        for j in range(stays):
            end = start + timedelta(days=5)
            last = j == stays - 1
            valid_to = "9999-12-31" if last and i % 4 else end.isoformat()
            conn.execute(
                "INSERT INTO WARD_STAY VALUES (?, ?, ?, ?, ?)",
                (f"S{pid}-{j}", pid, wards[(i + j) % 4], start.isoformat(), valid_to),
            )
            start = end  # the next stay starts the moment this one ends
        for d in range(5):
            conn.execute(
                "INSERT INTO DAILY_CENSUS VALUES (?, ?, ?)",
                (f"2024-03-0{d + 1}", pid, ["STABLE", "CRITICAL", "DISCHARGED"][(i + d) % 3]),
            )
    conn.commit()
    return conn


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    adapter = SqliteAdapter(_hospital())
    storage = Storage(tmp_path_factory.mktemp("data"))
    run_profile(adapter, storage, ProfileConfig(workers=1))
    run = run_detect(adapter, storage, DetectConfig(workers=1))
    return run, storage


def test_every_table_detected(learned):
    run, _ = learned
    assert all(t["status"] == "detected" for t in run["tables"].values())


def test_history_table_found_with_its_never_ends_date(learned):
    _, storage = learned
    finding = storage.read(FINDINGS, f"{SCHEMA}.WARD_STAY")
    assert finding["kind"] == "history"
    history = finding["history"]
    assert history["entity"] == ["PATIENT_ID"]
    assert (history["start"], history["end"]) == ("VALID_FROM", "VALID_TO")
    assert history["open_marker"]["end_value"] == "9999-12-31"
    assert history["boundary"] == "end_equals_next_start"
    assert history["evidence"]["overlapping_versions"] == 0
    assert finding["grain"]["columns"] == ["PATIENT_ID", "VALID_FROM"]


def test_repeated_daily_copies_found(learned):
    _, storage = learned
    finding = storage.read(FINDINGS, f"{SCHEMA}.DAILY_CENSUS")
    assert finding["kind"] == "snapshot"
    assert finding["snapshot"]["entity"] == ["PATIENT_ID"]
    assert finding["snapshot"]["date_column"] == "CENSUS_DAY"
    assert finding["snapshot"]["current_rule"]["value"] == "2024-03-05"


def test_plain_list_table_keyed_by_its_id(learned):
    _, storage = learned
    finding = storage.read(FINDINGS, f"{SCHEMA}.PATIENT")
    assert finding["kind"] == "keyed"
    assert finding["grain"]["columns"] == ["ID"]


def test_links_found_without_declared_keys(learned):
    _, storage = learned
    links = storage.read(LINKS, "links")["links"]
    best = {
        (l["child"]["table"], l["child"]["column"]): l["parent"]
        for l in links
        if l["status"] == "accepted" and l["best"]
    }
    patient = {"table": f"{SCHEMA}.PATIENT", "column": "ID"}
    assert best[(f"{SCHEMA}.WARD_STAY", "PATIENT_ID")] == patient
    assert best[(f"{SCHEMA}.DAILY_CENSUS", "PATIENT_ID")] == patient


def test_detect_is_repeatable(learned):
    _, storage = learned
    first = {k: storage.read(FINDINGS, k) for k in storage.keys(FINDINGS)}
    adapter = SqliteAdapter(_hospital())
    run_profile(adapter, storage, ProfileConfig(workers=1), force=True)
    run_detect(adapter, storage, DetectConfig(workers=1))
    assert {k: storage.read(FINDINGS, k) for k in storage.keys(FINDINGS)} == first


def test_name_score_reads_abbreviations_and_bare_ids():
    cfg = DetectConfig()
    assert name_score("SUBSCR_ID", "ANY", "SUBSCRIPTION_ID", cfg) == 1.0
    assert name_score("PATIENT_ID", "PATIENT", "ID", cfg) == 1.0
    assert name_score("CUSTOMER_ID", "ORDERS", "ORDER_ID", cfg) < cfg.link_min_name_score


def test_no_source_names_in_code():
    """Names learned from the real source must never appear in Rosetta's code."""
    profiles = Path("data") / "profiles"
    if not profiles.exists():
        pytest.skip("no stored profiles to compare against")
    import json

    names = set()
    for path in profiles.glob("*.json"):
        profile = json.loads(path.read_text(encoding="utf-8"))
        names.add(profile["table"])
        names |= {c["name"] for c in profile["columns"] if "_" in c["name"]}
    code = "\n".join(p.read_text(encoding="utf-8") for p in Path("rosetta").rglob("*.py"))
    found = sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", code))
    assert not found, f"source names hard-coded in rosetta/: {found}"
