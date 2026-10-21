"""HTTP front door. Thin: it calls the Learn and Answer modules and holds no logic."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from rosetta.adapters import get_adapter
from rosetta.config import Settings
from rosetta.learn.detect import run_detect
from rosetta.learn.profile import run_profile
from rosetta.storage import Storage

app = FastAPI(title="Rosetta", version="0.1.0")


class LearnRequest(BaseModel):
    stage: Literal["profile", "detect"] = "profile"
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
    storage = Storage(settings.data_dir)
    try:
        if request.stage == "detect":
            return run_detect(adapter, storage)
        return run_profile(adapter, storage, force=request.force)
    finally:
        adapter.close()
