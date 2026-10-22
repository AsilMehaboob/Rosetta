"""HTTP front door. Thin: it calls the Learn and Answer modules and holds no logic."""

import os
import threading
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rosetta.adapters import get_adapter
from rosetta.answer.extract import extract
from rosetta.answer.pipeline import answer_question
from rosetta.answer.retrieve import load_index, retrieve
from rosetta.answer.sql_agent import run_agent
from rosetta.learn.profile import PROFILES
from rosetta.config import Settings
from rosetta import evaluate
from rosetta.learn.annotate import run_annotate
from rosetta.learn.assemble import run_assemble
from rosetta.learn.detect import run_detect
from rosetta.learn.profile import run_profile
from rosetta.llm import GeminiLLM
from rosetta.storage import Storage

app = FastAPI(title="Rosetta", version="0.1.0")

# the browser UI runs on its own address; only the addresses listed here may call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


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


@app.post("/retrieve")
def retrieve_for_question(request: ExtractRequest) -> dict:
    """Show the slice of the semantic layer a question would be answered from. Nothing is queried."""
    settings = Settings.from_env()
    storage = Storage(settings.data_dir)
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, storage)
    reference = request.reference_date or settings.reference_date or date.today()
    extracted = extract(llm, request.question, reference)
    dialect = get_adapter(settings).sqlglot_dialect
    return {"extract": extracted, "retrieve": retrieve(load_index(storage), extracted, dialect)}


@app.post("/sql")
def sql_for_question(request: ExtractRequest) -> dict:
    """Extract, retrieve, then let the SQL agent write and run the query (read-only)."""
    settings = Settings.from_env()
    storage = Storage(settings.data_dir)
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, storage)
    reference = request.reference_date or settings.reference_date or date.today()
    index = load_index(storage)
    extracted = extract(llm, request.question, reference)
    adapter = get_adapter(settings)
    pack = retrieve(index, extracted, adapter.sqlglot_dialect)
    if pack["decision"] != "answer":
        return {"extract": extracted, "retrieve": pack, "agent": {"status": "cannot_answer", "reason": pack["reason"]}}
    profiles = {k: storage.read(PROFILES, k) for k in index.tables}
    adapter.connect_pool()
    try:
        agent = run_agent(llm, adapter, index, profiles, extracted, pack)
    finally:
        adapter.close()
    return {"extract": extracted, "retrieve": pack, "agent": agent}


@app.post("/questions")
def ask(request: ExtractRequest) -> dict:
    """Answer a business question: the answer, the rows, the query and the evidence behind them."""
    settings = Settings.from_env()
    storage = Storage(settings.data_dir)
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, storage)
    reference = request.reference_date or settings.reference_date or date.today()
    adapter = get_adapter(settings)
    try:
        return answer_question(llm, adapter, storage, load_index(storage), request.question, reference)
    finally:
        adapter.close()


# --- evaluation: run the benchmark in the background and read saved results


class EvalRequest(BaseModel):
    runs: int = Field(default=3, ge=1, le=5, description="times each question is asked")
    only: list[str] | None = Field(default=None, description="question ids to run; all when empty")
    reference_date: date | None = None


_eval_lock = threading.Lock()
_eval_state: dict = {"running": False}


def _run_eval(request: EvalRequest) -> None:
    def progress(state: dict) -> None:
        _eval_state.update(state)

    try:
        evaluate.run_benchmark(request.runs, request.only, request.reference_date, progress)
    except Exception as exc:  # shown in the status, so the UI can say why
        _eval_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _eval_state["running"] = False


@app.get("/eval/benchmark")
def eval_benchmark() -> list[dict]:
    """The benchmark questions (without their reference queries)."""
    return [{k: v for k, v in q.items() if k != "reference_sql"} for q in evaluate.load_benchmark()["questions"]]


@app.post("/eval/run")
def eval_run(request: EvalRequest) -> dict:
    """Start an evaluation in the background; follow it with /eval/status."""
    with _eval_lock:
        if _eval_state.get("running"):
            raise HTTPException(409, "An evaluation is already running")
        _eval_state.clear()
        _eval_state.update({"running": True, "done": 0, "total": None, "error": None})
    threading.Thread(target=_run_eval, args=(request,), daemon=True).start()
    return dict(_eval_state)


@app.get("/eval/status")
def eval_status() -> dict:
    return dict(_eval_state)


@app.get("/eval/results")
def eval_results() -> list[dict]:
    """Saved evaluations, newest first."""
    return evaluate.list_results()


@app.get("/eval/results/{run_id}")
def eval_result(run_id: str) -> dict:
    """One saved evaluation: summary, every run and the expected rows."""
    found = evaluate.read_result(run_id)
    if found is None:
        raise HTTPException(404, "No such evaluation")
    return found
