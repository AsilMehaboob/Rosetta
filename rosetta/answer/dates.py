"""Turn date phrases into exact date ranges, in plain code.

The LLM only copies date phrases out of a question; this module decides what
they mean, so the same phrase is always read the same way and every reading is
written down as a sentence for the answer. Periods are half-open: a month runs
from its first day up to, not including, the first day of the next month.
"""

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

_MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ],
        1,
    )
    for name in names
}
_MONTH = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(\d{4})"
_UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month",
          "months": "month", "year": "year", "years": "year"}
_POINT_PREFIX = re.compile(r"^(as of|as at|as on|at|on|by|until the end of)\s+")
_FILLER = re.compile(r"^(in|during|for|over|within|from|of|the)\s+")


@dataclass(frozen=True)
class Period:
    start: date
    end: date  # exclusive

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def _month(year: int, month: int) -> Period:
    start = date(year, month, 1)
    return Period(start, _add_months(start, 1))


def _add_months(d: date, n: int) -> date:
    index = d.year * 12 + d.month - 1 + n
    year, month = divmod(index, 12)
    return date(year, month + 1, min(d.day, monthrange(year, month + 1)[1]))


def _fmt(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def _describe(p: Period) -> str:
    return f"{_fmt(p.start)} up to (not including) {_fmt(p.end)}"


def _parse(text: str, ref: date) -> list[Period] | date | None:
    """A list of periods, a single point in time, or None when the phrase is not understood."""
    t = re.sub(r"\s+", " ", text.lower().strip().strip(".,?"))
    t = t.replace(",", " ").replace("  ", " ")

    if t in ("today", "now", "currently", "current", "present", "at present", "right now", "as of now",
             "as of today", "to date", "so far"):
        return ref
    if t == "yesterday":
        return ref - timedelta(days=1)

    relative = re.fullmatch(r"(this|current|last|previous|past|prior) (month|year|quarter|week)", t)
    if relative:
        which, unit = relative.groups()
        back = 0 if which in ("this", "current") else 1
        if unit == "month":
            start = _add_months(date(ref.year, ref.month, 1), -back)
            return [Period(start, _add_months(start, 1))]
        if unit == "year":
            return [Period(date(ref.year - back, 1, 1), date(ref.year - back + 1, 1, 1))]
        if unit == "week":
            monday = ref - timedelta(days=ref.weekday()) - timedelta(weeks=back)
            return [Period(monday, monday + timedelta(weeks=1))]
        q_start = _add_months(date(ref.year, 3 * ((ref.month - 1) // 3) + 1, 1), -3 * back)
        return [Period(q_start, _add_months(q_start, 3))]

    last_n = re.fullmatch(r"(?:the )?(?:last|past|previous) (\d+) (days?|weeks?|months?|years?)", t)
    if last_n:
        n, unit = int(last_n.group(1)), _UNITS[last_n.group(2)]
        end = ref + timedelta(days=1)
        start = {
            "day": end - timedelta(days=n),
            "week": end - timedelta(weeks=n),
            "month": _add_months(end, -n),
            "year": _add_months(end, -12 * n),
        }[unit]
        return [Period(start, end)]

    between = re.fullmatch(r"(?:between |from )?(.+?) (?:and|to|through|until|-) (.+)", t)
    if between and not re.fullmatch(rf"{_MONTH}(?: (?:and|&) {_MONTH})+ {_YEAR}", t):
        left, right = _parse(between.group(1), ref), _parse(between.group(2), ref)
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 1:
            if left[0].start < right[0].end:
                return [Period(left[0].start, right[0].end)]

    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if iso:
        return date(*map(int, iso.groups()))
    day_month = re.fullmatch(rf"{_DAY} (?:of )?{_MONTH} {_YEAR}", t)
    if day_month:
        return date(int(day_month.group(3)), _MONTHS[day_month.group(2)], int(day_month.group(1)))
    month_day = re.fullmatch(rf"{_MONTH} {_DAY} {_YEAR}", t)
    if month_day:
        return date(int(month_day.group(3)), _MONTHS[month_day.group(1)], int(month_day.group(2)))

    months = re.fullmatch(rf"{_MONTH}((?: (?:and|&) {_MONTH})*) {_YEAR}", t)
    if months:
        year = int(months.group(months.lastindex))
        names = [months.group(1)] + re.findall(_MONTH, months.group(2))
        return [_month(year, _MONTHS[n]) for n in names]

    quarter = re.fullmatch(rf"q([1-4]) {_YEAR}|{_YEAR} q([1-4])", t)
    if quarter:
        q = int(quarter.group(1) or quarter.group(4))
        year = int(quarter.group(2) or quarter.group(3))
        start = date(year, 3 * (q - 1) + 1, 1)
        return [Period(start, _add_months(start, 3))]

    year_only = re.fullmatch(_YEAR, t)
    if year_only:
        year = int(year_only.group(1))
        return [Period(date(year, 1, 1), date(year + 1, 1, 1))]
    return None


def resolve(text: str, ref: date) -> dict[str, Any]:
    """Read one date phrase against a reference date ("today")."""
    stripped = text.strip()
    lowered = stripped.lower()
    point_in_time = bool(_POINT_PREFIX.match(lowered))
    core = _POINT_PREFIX.sub("", lowered)
    while _FILLER.match(core):
        core = _FILLER.sub("", core)

    parsed = _parse(core, ref)
    if parsed is None:
        return {"text": stripped, "kind": "unresolved", "reading": f"could not read '{stripped}' as a date"}
    if isinstance(parsed, date):
        return {
            "text": stripped,
            "kind": "point",
            "at": parsed.isoformat(),
            "reading": f"'{stripped}' read as the state on {_fmt(parsed)}",
        }
    if point_in_time and len(parsed) == 1:  # "as of June 2025": the state at the end of that period
        last_day = parsed[0].end - timedelta(days=1)
        return {
            "text": stripped,
            "kind": "point",
            "at": last_day.isoformat(),
            "reading": f"'{stripped}' read as the state on {_fmt(last_day)}, the last day of that period",
        }
    readings = "; ".join(_describe(p) for p in parsed)
    return {
        "text": stripped,
        "kind": "period",
        "periods": [p.as_dict() for p in parsed],
        "reading": f"'{stripped}' read as {readings}",
    }
