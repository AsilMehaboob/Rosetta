"""One question from start to finish: extract, retrieve, write and run the query, explain the result.

With earlier turns of a conversation, a follow-up ("now split that by city") is first rewritten into one
complete question (converse.py), and the previous query is handed to the SQL agent as a starting point.

The API and the evaluation harness both answer through here, so what is scored is exactly what users get.
"""

import time
from datetime import date

from rosetta.adapters.base import SourceAdapter
from rosetta.answer.answer import build_answer
from rosetta.answer.converse import resolve
from rosetta.answer.extract import extract
from rosetta.answer.retrieve import LayerIndex, retrieve
from rosetta.answer.sql_agent import run_agent
from rosetta.learn.profile import PROFILES
from rosetta.llm import JsonLLM
from rosetta.storage import Storage


def answer_question(
    llm: JsonLLM,
    adapter: SourceAdapter,
    storage: Storage,
    index: LayerIndex,
    question: str,
    reference: date,
    history: list[dict] | None = None,
) -> dict:
    """The full answer for one question, with how long it took and how many model calls it made.

    history: earlier turns of the conversation, oldest first, each {question, sql}.
    """
    started = time.perf_counter()
    calls, cached = llm.calls, llm.cached
    turn = resolve(llm, question, history or [])
    extracted = extract(llm, turn["question"], reference)
    if turn["follow_up"] and history and history[-1].get("sql"):
        extracted["previous_query"] = {"question": history[-1]["question"], "sql": history[-1]["sql"]}
    pack = retrieve(index, extracted, adapter.sqlglot_dialect)
    agent = None
    if pack["decision"] == "answer":
        profiles = {k: storage.read(PROFILES, k) for k in index.tables}
        agent = run_agent(llm, adapter, index, profiles, extracted, pack)
    result = build_answer(llm, extracted, pack, agent, index.version)
    result["asked"] = question.strip()  # the message as typed; "question" is the complete question answered
    result["follow_up"] = turn["follow_up"]
    if turn["issue"]:
        result.setdefault("technical_notes", []).append(turn["issue"])
    result["seconds"] = round(time.perf_counter() - started, 2)
    result["llm_calls"] = llm.calls - calls
    result["llm_cached"] = llm.cached - cached
    return result
