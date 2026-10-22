"""HTTP front door. Thin: it calls the Learn and Answer modules and holds no logic."""

from datetime import date
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rosetta.adapters import get_adapter
from rosetta.answer.extract import extract
from rosetta.config import Settings
from rosetta.learn.annotate import run_annotate
from rosetta.learn.assemble import run_assemble
from rosetta.learn.detect import run_detect
from rosetta.learn.profile import run_profile
from rosetta.llm import GeminiLLM
from rosetta.storage import Storage

app = FastAPI(title="Rosetta", version="0.1.0")


class LearnRequest(BaseModel):
    stage: Literal["profile", "detect", "annotate", "assemble"] = "profile"
    force: bool = False  # re-measure even when a table's fingerprint is unchanged


class ExtractRequest(BaseModel):
    question: str = Field(min_length=1, examples=["How many overdue invoices were there in Berlin last month?"])
    reference_date: date | None = Field(
        default=None, description='"Today" for relative dates such as "last month"; defaults to REFERENCE_DATE or today'
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/learn")
def learn(request: LearnRequest) -> dict:
    """Run one Learn stage over the configured source database."""
    settings = Settings.from_env()
    adapter = get_adapter(settings)
    storage = Storage(settings.data_dir)
    if request.stage == "annotate":  # works from stored evidence only, never the source
        llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, storage)
        return run_annotate(llm, storage, adapter.sqlglot_dialect, adapter.dialect_rules())
    adapter.connect_pool()
    try:
        if request.stage == "detect":
            return run_detect(adapter, storage)
        if request.stage == "assemble":
            return run_assemble(adapter, storage)
        return run_profile(adapter, storage, force=request.force)
    finally:
        adapter.close()


@app.post("/extract")
def extract_question(request: ExtractRequest) -> dict:
    """Show how a question is read: its business words, values and dates. Nothing is queried."""
    settings = Settings.from_env()
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, Storage(settings.data_dir))
    reference = request.reference_date or settings.reference_date or date.today()
    return extract(llm, request.question, reference)
