"""HTTP front door. Thin: it calls the Learn and Answer modules and holds no logic."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from rosetta.adapters import get_adapter
from rosetta.config import Settings
from rosetta.learn.profile import run_profile
from rosetta.storage import Storage

app = FastAPI(title="Rosetta", version="0.1.0")


class LearnRequest(BaseModel):
    stage: Literal["profile"] = "profile"
    force: bool = False  # re-measure even when a table's fingerprint is unchanged


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/learn")
def learn(request: LearnRequest) -> dict:
    """Run one Learn stage over the configured source database."""
    settings = Settings.from_env()
    adapter = get_adapter(settings)
    adapter.connect_pool()
    try:
        return run_profile(adapter, Storage(settings.data_dir), force=request.force)
    finally:
        adapter.close()
