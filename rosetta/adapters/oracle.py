"""Oracle adapter (python-oracledb thin mode, no client install needed)."""

from datetime import date
from typing import Any, Sequence

import oracledb

from rosetta.adapters.base import Column, Key, QueryResult, SourceAdapter
from rosetta.config import Settings

_PLAN_STATEMENT_ID = "ROSETTA_COST"


class OracleAdapter(SourceAdapter):
    sqlglot_dialect = "oracle"

    def __init__(self, settings: Settings):
        self._s = settings
        self._pool: oracledb.ConnectionPool | None = None

    # connection
    def connect_pool(self) -> None:
        if self._pool is not None:
            return
        self._pool = oracledb.create_pool(
            user=self._s.user,
            password=self._s.password,
            dsn=f"{self._s.host}:{self._s.port}/{self._s.service}",
            min=1,
            max=4,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def run_select(
        self,
        sql: str,
        binds: dict[str, Any] | Sequence[Any] | None = None,
        timeout: int | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        self.connect_pool()
        timeout = timeout or self._s.query_timeout_seconds
        max_rows = max_rows or self._s.query_max_rows
        with self._pool.acquire() as conn:
            conn.call_timeout = timeout * 1000
            with conn.cursor() as cur:
                cur.execute(sql, binds or [])
                columns = tuple(d[0] for d in cur.description)
                rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return QueryResult(columns, tuple(tuple(r) for r in rows[:max_rows]), truncated)

    # catalog
    def _schema_binds(self) -> tuple[str, dict[str, str]]:
        names = {f"s{i}": s for i, s in enumerate(self._s.schemas)}
        return ", ".join(f":{k}" for k in names), names

    def list_tables(self) -> list[tuple[str, str]]:
        placeholders, binds = self._schema_binds()
        result = self.run_select(
            f"SELECT owner, table_name FROM all_tables WHERE owner IN ({placeholders}) "
            "ORDER BY owner, table_name",
            binds,
            max_rows=100_000,
        )
        return [(r[0], r[1]) for r in result.rows]

    def list_columns(self) -> list[Column]:
        placeholders, binds = self._schema_binds()
        result = self.run_select(
            "SELECT c.owner, c.table_name, c.column_name, c.data_type, c.data_length, "
            "c.data_precision, c.data_scale, c.nullable, c.column_id, m.comments "
            "FROM all_tab_columns c "
            "JOIN all_tables t ON t.owner = c.owner AND t.table_name = c.table_name "
            "LEFT JOIN all_col_comments m ON m.owner = c.owner "
            "AND m.table_name = c.table_name AND m.column_name = c.column_name "
            f"WHERE c.owner IN ({placeholders}) "
            "ORDER BY c.owner, c.table_name, c.column_id",
            binds,
            max_rows=100_000,
        )
        return [
            Column(
                schema=r[0],
                table=r[1],
                name=r[2],
                data_type=_format_type(r[3], r[4], r[5], r[6]),
                nullable=r[7] == "Y",
                position=r[8],
                comment=r[9],
            )
            for r in result.rows
        ]

    def list_keys(self) -> list[Key]:
        placeholders, binds = self._schema_binds()
        constraints = self.run_select(
            "SELECT owner, table_name, constraint_name, constraint_type, r_owner, "
            "r_constraint_name FROM all_constraints "
            f"WHERE constraint_type IN ('P', 'R') AND owner IN ({placeholders}) "
            "ORDER BY owner, table_name, constraint_name",
            binds,
            max_rows=100_000,
        )
        # every constraint's columns in order, including the referenced side
        cols = self.run_select(
            "SELECT owner, constraint_name, column_name FROM all_cons_columns "
            f"WHERE owner IN ({placeholders}) ORDER BY owner, constraint_name, position",
            binds,
            max_rows=1_000_000,
        )
        by_constraint: dict[tuple[str, str], list[str]] = {}
        for owner, name, column in cols.rows:
            by_constraint.setdefault((owner, name), []).append(column)

        referenced_tables = {
            (r[0], r[2]): r[1]
            for r in self.run_select(
                "SELECT owner, table_name, constraint_name FROM all_constraints "
                f"WHERE constraint_type IN ('P', 'U') AND owner IN ({placeholders})",
                binds,
                max_rows=1_000_000,
            ).rows
        }
        keys = []
        for owner, table, name, kind, r_owner, r_name in constraints.rows:
            columns = tuple(by_constraint.get((owner, name), []))
            if kind == "P":
                keys.append(Key(owner, table, name, "primary", columns))
            else:
                keys.append(
                    Key(
                        owner,
                        table,
                        name,
                        "foreign",
                        columns,
                        ref_schema=r_owner,
                        ref_table=referenced_tables.get((r_owner, r_name)),
                        ref_columns=tuple(by_constraint.get((r_owner, r_name), [])),
                    )
                )
        return keys

    def table_row_counts(self) -> dict[tuple[str, str], int | None]:
        placeholders, binds = self._schema_binds()
        result = self.run_select(
            f"SELECT owner, table_name, num_rows FROM all_tables WHERE owner IN ({placeholders})",
            binds,
            max_rows=100_000,
        )
        return {(r[0], r[1]): (int(r[2]) if r[2] is not None else None) for r in result.rows}

    # syntax pieces
    def quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def limit_clause(self, n: int) -> str:
        return f"FETCH FIRST {int(n)} ROWS ONLY"

    def date_literal(self, value: date) -> str:
        return f"DATE '{value.isoformat()}'"

    # profiling helpers
    def approx_distinct_expr(self, column: str) -> str:
        return f"APPROX_COUNT_DISTINCT({self.quote(column)})"

    def sample_clause(self, percent: float, seed: int) -> str:
        return f"SAMPLE ({percent}) SEED ({int(seed)})"

    # safety
    def explain_cost(self, sql: str) -> float | None:
        self.connect_pool()
        try:
            with self._pool.acquire() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"EXPLAIN PLAN SET STATEMENT_ID = '{_PLAN_STATEMENT_ID}' FOR {sql}"
                    )
                    cur.execute(
                        "SELECT cost FROM plan_table WHERE statement_id = :id AND id = 0",
                        id=_PLAN_STATEMENT_ID,
                    )
                    row = cur.fetchone()
                    conn.rollback()
            return float(row[0]) if row and row[0] is not None else None
        except oracledb.DatabaseError:
            return None

    def map_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        if "ORA-00942" in message:
            return f"unknown_table: {message}"
        if "ORA-00904" in message:
            return f"unknown_column: {message}"
        if "DPY-4024" in message or "ORA-01013" in message or "timeout" in message.lower():
            return f"timeout: {message}"
        if "ORA-009" in message:
            return f"syntax: {message}"
        return f"error: {message}"

    # prompt rules
    def dialect_rules(self) -> str:
        return (
            "The database is Oracle. Limit rows with FETCH FIRST n ROWS ONLY, never LIMIT. "
            "Write date literals as DATE 'YYYY-MM-DD'. Quote identifiers with double quotes "
            "only when they are mixed case. Do not end the statement with a semicolon. "
            "Use TRUNC(date, 'MM') to get the first day of a month and ADD_MONTHS to move "
            "between months. An empty string is NULL in Oracle."
        )


def _format_type(
    data_type: str, length: int | None, precision: int | None, scale: int | None
) -> str:
    if data_type == "NUMBER" and precision is not None:
        return f"NUMBER({precision},{scale or 0})"
    if data_type in ("VARCHAR2", "CHAR", "NVARCHAR2", "NCHAR") and length:
        return f"{data_type}({length})"
    return data_type
