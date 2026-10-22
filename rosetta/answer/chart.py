"""Chart: suggest the one chart that tells a result's story, or none when a chart would add nothing.

Decided by code from the shape of the result, never from table or column names, so it works for
any database: which columns hold periods (dates or months), which hold labels, which hold figures.
A chart is suggested only when there is something to see: at least two rows, a figure that varies,
and a result that is not a plain list. Running changes (a column that starts empty, such as the
change from the previous period) stay in the table; the chart shows the figures themselves.
"""

import re
from dataclasses import dataclass
from typing import Any

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class ChartConfig:
    line_min_periods: int = 4  # fewer periods read better as side-by-side bars
    max_bars: int = 15  # longer rankings show their top bars, with a note
    max_rows: int = 50  # beyond this the table tells the story better
    max_grouped_rows: int = 12
    max_series: int = 4
    share_max_parts: int = 6  # a few parts of one whole are shown with their share


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _label(column: str) -> str:
    words = column.replace("_", " ").strip().lower()
    return words[:1].upper() + words[1:]


def suggest_chart(columns: list[str], rows: list[list], extracted: dict, cfg: ChartConfig | None = None) -> dict | None:
    """The chart for this result, or None. The data is passed through as rows of {x, series...}."""
    cfg = cfg or ChartConfig()
    if len(rows) < 2 or len(rows) > cfg.max_rows or extracted.get("output") == "list":
        return None

    def values(j: int) -> list:
        return [r[j] for r in rows]

    times = [j for j in range(len(columns)) if all(isinstance(v, str) and (_DAY.match(v) or _MONTH.match(v)) for v in values(j))]
    figures = [
        j for j in range(len(columns))
        if j not in times and all(_is_number(v) for v in values(j)) and len(set(values(j))) > 1
    ]
    labels = [j for j in range(len(columns)) if j not in times and not all(_is_number(v) or v is None for v in values(j))]
    if not figures:
        return None
    figures = figures[: cfg.max_series]

    def point(r: list, x: int) -> dict:
        return {"x": r[x], **{columns[j]: r[j] for j in figures}}

    series = [{"key": columns[j], "label": _label(columns[j])} for j in figures]
    what = " and ".join(s["label"].lower() for s in series)

    if times:
        x = times[0]
        if len({r[x] for r in rows}) != len(rows):
            return None  # several rows per period: a table reads better
        data = [point(r, x) for r in sorted(rows, key=lambda r: r[x])]
        kind = "line" if len(rows) >= cfg.line_min_periods else "column"
        # named by what the values are, not by the column: month starts are months
        unit = "month" if all(_MONTH.match(r[x]) or r[x][8:10] == "01" for r in rows) else "date"
        return {
            "type": kind,
            "x": {"key": columns[x], "label": _label(columns[x]), "time": True},
            "series": series,
            "data": data,
            "title": f"{what[:1].upper()}{what[1:]} by {unit}",
            "reason": (
                "A line shows how the figures move over time" if kind == "line"
                else "Bars side by side compare the periods directly"
            ),
            "note": None,
        }

    if not labels:
        return None
    x = labels[0]
    if len({r[x] for r in rows}) != len(rows):
        return None  # the label repeats: the rows are split further than one chart can show
    if len(figures) > 1 and len(rows) <= cfg.max_grouped_rows:
        return {
            "type": "column",
            "x": {"key": columns[x], "label": _label(columns[x]), "time": False},
            "series": series,
            "data": [point(r, x) for r in rows],
            "title": f"{what[:1].upper()}{what[1:]} by {_label(columns[x]).lower()}",
            "reason": "Grouped bars compare several figures for each item",
            "note": None,
        }

    figure = figures[0]
    series = series[:1]
    ordered = sorted(rows, key=lambda r: -(r[figure] or 0))
    shown = ordered[: cfg.max_bars]
    total = sum(r[figure] for r in rows)
    parts = len(rows) <= cfg.share_max_parts and all(r[figure] >= 0 for r in rows) and total > 0
    data = [point(r, x) for r in shown]
    if parts:
        for d, r in zip(data, shown):
            d["share"] = round(r[figure] / total * 100, 1)
    return {
        "type": "share" if parts else "bar",
        "x": {"key": columns[x], "label": _label(columns[x]), "time": False},
        "series": series,
        "data": data,
        "title": f"{series[0]['label']} by {_label(columns[x]).lower()}",
        "reason": (
            "Each bar shows a part of the whole and its share" if parts
            else "Bars sorted from highest to lowest show the ranking"
        ),
        "note": f"Showing the top {len(shown)} of {len(rows)}" if len(shown) < len(rows) else None,
    }
