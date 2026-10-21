"""Connect through the adapter and list what the catalog exposes. Prints no credentials."""

from collections import Counter

from rosetta.adapters import get_adapter
from rosetta.config import Settings


def main() -> None:
    settings = Settings.from_env()
    adapter = get_adapter(settings)
    adapter.connect_pool()
    try:
        print(f"connected: {settings.user}@{settings.host}:{settings.port}/{settings.service}")
        tables = adapter.list_tables()
        columns = adapter.list_columns()
        keys = adapter.list_keys()
        counts = adapter.table_row_counts()
        per_table = Counter((c.schema, c.table) for c in columns)
        print(f"{len(tables)} tables, {len(columns)} columns, {len(keys)} declared keys")
        for schema, table in tables:
            rows = counts.get((schema, table))
            print(f"  {schema}.{table}: {per_table[(schema, table)]} columns, rows={rows}")
        print("declared keys by kind:", dict(Counter(k.kind for k in keys)))
        probe = adapter.run_select("SELECT 1 AS ok FROM dual")
        print("probe query:", probe.columns, probe.rows)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
