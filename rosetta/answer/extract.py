"""Extract: copy the business words, values and dates out of a question.

The LLM only copies phrases out of the question into slots; it does not
interpret them or name tables. Code then keeps only phrases that really occur
in the question, and reads the date phrases itself (see dates.py), so the
same question is always read the same way.
"""

import re
from datetime import date
from typing import Any

from rosetta.answer.dates import resolve
from rosetta.llm import JsonLLM

PROMPT_VERSION = "extract-2"

_SLOTS = ("subjects", "conditions", "values", "measures", "group_by", "dates")

SCHEMA = {
    "type": "object",
    "properties": {
        **{slot: {"type": "array", "items": {"type": "string"}} for slot in _SLOTS},
        "output": {"type": "string", "enum": ["list", "count", "sum", "average", "max", "min", "compare", "other"]},
    },
    "required": [*_SLOTS, "output"],
}

SYSTEM = """You read one business question and copy its parts into slots. Copy words exactly as they
appear in the question. Do not interpret, translate, correct or explain them, and never name database
tables or columns.
- subjects: the things being asked about (for example "patients", "invoices", "flights").
- conditions: words that narrow those things down (for example "overdue", "first-time", "cancelled", "distinct").
- values: names, places, codes or numbers that must match the data exactly (for example a product code).
- measures: what is to be calculated (for example "how many", "total revenue", "average delay").
- group_by: what the result is split by (for example "per region", "by product line", "for each week").
- dates: every date or time phrase, including "as of ..." and relative ones like "last month" or "currently".
- output: the shape of the answer: list, count, sum, average, max, min, compare or other.
A phrase may appear in more than one slot. Leave a slot empty when the question has nothing for it."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _occurs(phrase: str, question: str) -> bool:
    """True when the phrase, or every word of it, is in the question."""
    p, q = _norm(phrase), _norm(question)
    if p and p in q:
        return True
    words = re.findall(r"\w+", p)
    return bool(words) and all(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


def extract(llm: JsonLLM, question: str, reference_date: date) -> dict[str, Any]:
    raw = llm.generate_json(system=SYSTEM, prompt=question.strip(), schema=SCHEMA, prompt_version=PROMPT_VERSION)

    issues, slots = [], {}
    for slot in _SLOTS:
        kept = []
        for phrase in raw.get(slot, []):
            phrase = phrase.strip()
            if not phrase or phrase.casefold() in {k.casefold() for k in kept}:
                continue  # the same phrase again, perhaps in other capitals
            if _occurs(phrase, question):
                kept.append(phrase)
            else:  # the model added something that is not in the question
                issues.append(f"dropped {slot} phrase {phrase!r}: not in the question")
        slots[slot] = kept

    dates = [resolve(d, reference_date) for d in slots["dates"]]
    issues += [d["reading"] for d in dates if d["kind"] == "unresolved"]
    return {
        "question": question.strip(),
        "reference_date": reference_date.isoformat(),
        **{slot: slots[slot] for slot in _SLOTS if slot != "dates"},
        "output": raw.get("output", "other"),
        "dates": dates,
        "issues": issues,
    }
