"""The contract every source-database engine implements.

Everything engine-specific lives behind this interface so the Learn and Answer
modules never see a driver, a catalog view or a SQL dialect quirk.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence


@dataclass(frozen=True)
class Column:
    schema: str
    table: str
    name: str
    data_type: str
    nullable: bool
    position: int
    comment: str | None = None


@dataclass(frozen=True)
class Key:
    schema: str
    table: str
    name: str
    kind: str  # "primary" or "foreign"
    columns: tuple[str, ...]
    ref_schema: str | None = None
    ref_table: str | None = None
    ref_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool  # True when max_rows cut the result short


class SourceAdapter(ABC):
    sqlglot_dialect: str

    # connection
    @abstractmethod
    def connect_pool(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def run_select(
        self,
        sql: str,
        binds: dict[str, Any] | Sequence[Any] | None = None,
        timeout: int | None = None,
        max_rows: int | None = None,
    ) -> QueryResult: ...

    # catalog
    @abstractmethod
    def list_tables(self) -> list[tuple[str, str]]: ...

    @abstractmethod
    def list_columns(self) -> list[Column]: ...

    @abstractmethod
    def list_keys(self) -> list[Key]: ...

    @abstractmethod
    def table_row_counts(self) -> dict[tuple[str, str], int | None]: ...

    # syntax pieces
    @abstractmethod
    def quote(self, identifier: str) -> str: ...

    @abstractmethod
    def limit_clause(self, n: int) -> str: ...

    @abstractmethod
    def date_literal(self, value: date) -> str: ...

    @abstractmethod
    def datetime_literal(self, value: datetime) -> str:
        """A literal that keeps the time of day, so it equals a stored value exactly."""

    @abstractmethod
    def type_family(self, data_type: str) -> str:
        """Map an engine type name to "date", "number", "text" or "other"."""

    # profiling helpers
    @abstractmethod
    def approx_distinct_expr(self, column: str) -> str: ...

    @abstractmethod
    def sample_clause(self, percent: float, seed: int) -> str: ...

    # safety
    @abstractmethod
    def explain_cost(self, sql: str) -> float | None: ...

    @abstractmethod
    def map_error(self, exc: Exception) -> str: ...

    # prompt rules
    @abstractmethod
    def dialect_rules(self) -> str: ...
