"""Converse: turn a follow-up ("now split that by city") into one complete question.

Only runs when the conversation has earlier questions. The LLM decides whether the new
message stands on its own or builds on the previous question, and for a follow-up writes
one complete question. Code then checks the rewrite uses only words the user has already
written (in this message or the earlier questions), so it cannot add a condition, value or
date of its own; if it does, a plain combination of the two is used instead. The rest of the
pipeline answers the complete question exactly as if it had been asked directly.
"""

import json
import re
from typing import Any

from rosetta.answer.retrieve import fold
from rosetta.llm import JsonLLM

PROMPT_VERSION = "converse-1"
MAX_TURNS = 3  # earlier questions considered

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["standalone", "follow_up"]},
        "question": {"type": "string"},
    },
    "required": ["kind", "question"],
}

SYSTEM = """You read the new message in a conversation about data and decide whether it is a new question or a
follow-up to the previous one.
- standalone: the message makes sense on its own. Return it unchanged as the question.
- follow_up: the message builds on the previous question ("that", "those", "now split by ...", "what about ...",
  "and for March?", "only the ..."). Write ONE complete question that keeps everything from the previous question
  (what is counted, its conditions, values and dates) and applies the change the message asks for: add a split,
  replace a value or a date, add or remove a condition.
Rules:
1. Use only words from the earlier questions and the new message. Add nothing of your own: no new conditions,
   values, dates, names or business terms.
2. Keep the user's own words; do not rephrase or explain them.
3. When the message replaces something (another place, another month), drop what it replaces.
Examples (from any business):
- previous "How many overdue invoices were there in Berlin last month?", message "now split that by customer type"
  -> "How many overdue invoices were there in Berlin last month, split by customer type?"
- previous "Total revenue per region in 2024", message "what about 2023?" -> "Total revenue per region in 2023"
- message "Which products sold the most in May?" -> standalone"""

# small joining words a rewrite may need although the user did not type them
_JOINERS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "by", "per", "each", "with", "to", "at", "from", "split",
    "how", "many", "much", "what", "which", "were", "was", "is", "are", "there", "only", "also", "or",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", fold(text))


def resolve(llm: JsonLLM, message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """{question, follow_up, issue}: the complete question to answer for this message."""
    message = message.strip()
    previous = [h["question"].strip() for h in history if h.get("question", "").strip()][-MAX_TURNS:]
    if not previous:
        return {"question": message, "follow_up": False, "issue": None}
    prompt = json.dumps({"earlier_questions": previous, "new_message": message}, ensure_ascii=False)
    try:
        out = llm.generate_json(system=SYSTEM, prompt=prompt, schema=SCHEMA, prompt_version=PROMPT_VERSION)
    except Exception as exc:  # without the rewrite the message is still answered on its own
        return {"question": message, "follow_up": False, "issue": f"could not read the follow-up: {exc}"}
    if out.get("kind") != "follow_up" or not out.get("question", "").strip():
        return {"question": message, "follow_up": False, "issue": None}

    rewrite = out["question"].strip()
    allowed = set(_words(message)) | {w for q in previous for w in _words(q)} | _JOINERS
    added = sorted({w for w in _words(rewrite) if w not in allowed})
    if added:
        # the rewrite brought in words the user never wrote; fall back to the user's own words only
        return {
            "question": f"{previous[-1].rstrip('?. ')}; {message}",
            "follow_up": True,
            "issue": f"the rewrite added {', '.join(added)}; the two questions were combined as written instead",
        }
    return {"question": rewrite, "follow_up": True, "issue": None}
