from rosetta.adapters.base import SourceAdapter
from rosetta.config import Settings


def get_adapter(settings: Settings) -> SourceAdapter:
    """Pick the adapter for the configured dialect. One file per engine."""
    if settings.dialect == "oracle":
        from rosetta.adapters.oracle import OracleAdapter

        return OracleAdapter(settings)
    raise ValueError(f"No adapter for dialect {settings.dialect!r}")
