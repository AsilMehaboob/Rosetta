"""One question from start to finish: extract, retrieve, write and run the query, explain the result.

The API and the evaluation harness both answer through here, so what is scored is exactly what users get.
"""

import time
from datetime import date

from rosetta.adapters.base import SourceAdapter
from rosetta.answer.answer import build_answer
from rosetta.answer.extract import extract
from rosetta.answer.retrieve import LayerIndex, retrieve
from rosetta.answer.sql_agent import run_agent
from rosetta.learn.profile import PROFILES
from rosetta.llm import JsonLLM
from rosetta.storage import Storage


def answer_question(
    llm: JsonLLM, adapter: SourceAdapter, storage: Storage, index: LayerIndex, question: str, reference: date
) -> dict:
    """The full answer for one question, with how long it took and how many model calls it made."""
    started = time.perf_counter()
    calls, cached = llm.calls, llm.cached
    extracted = extract(llm, question, reference)
    pack = retrieve(index, extracted, adapter.sqlglot_dialect)
    agent = None
    if pack["decision"] == "answer":
        profiles = {k: storage.read(PROFILES, k) for k in index.tables}
        agent = run_agent(llm, adapter, index, profiles, extracted, pack)
    result = build_answer(llm, extracted, pack, agent, index.version)
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["llm_calls"] = llm.calls - calls
    result["llm_cached"] = llm.cached - cached
    return result
