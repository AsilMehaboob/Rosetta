"""Settings read from the environment (.env in development). No secrets live in code."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting {name}; see .env.example")
    return value


@dataclass(frozen=True)
class Settings:
    dialect: str
    host: str
    port: int
    service: str
    user: str
    password: str = field(repr=False)
    schemas: tuple[str, ...]
    query_timeout_seconds: int
    query_max_rows: int
    data_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        schemas = tuple(
            sorted(s.strip().upper() for s in _required("SOURCE_SCHEMAS").split(",") if s.strip())
        )
        return cls(
            dialect=os.getenv("SOURCE_DIALECT", "oracle").strip().lower(),
            host=_required("ORACLE_HOST"),
            port=int(os.getenv("ORACLE_PORT", "1521")),
            service=_required("ORACLE_SERVICE"),
            user=_required("ORACLE_USER"),
            password=_required("ORACLE_PASSWORD"),
            schemas=schemas,
            query_timeout_seconds=int(os.getenv("QUERY_TIMEOUT_SECONDS", "30")),
            query_max_rows=int(os.getenv("QUERY_MAX_ROWS", "1000")),
            data_dir=os.getenv("DATA_DIR", "data"),
        )
