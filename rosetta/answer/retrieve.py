"""Retrieve: find the slice of the semantic layer a question needs.

Plain code, no LLM. The phrases Extract copied out are matched three ways:
by words against terms, tables and columns (BM25 with accent folding and
abbreviation matching), and by value against the values seen in the data, so
"Zurich" finds the column that holds it and its exact stored spelling.
Tables are ranked from those matches, joined along verified links only, and
packed with their current-row rules and every double-counting warning. When
nothing matches, the pack says the question cannot be answered and why.
"""

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rosetta.learn.assemble import LAYERS
from rosetta.learn.profile import PROFILES
from rosetta.storage import Storage

# words that carry no meaning for finding data
_STOP = set(
    """a an the of in on at to for by with and or from as is are was were be been had has have do does did
    how many much which what who whose whom where when list show give tell me their its his her our your this
    that these those there each every per all any some into than then""".split()
)
# in this order: names and subjects pick the main tables, the rest prefer tables already chosen
_SLOT_WEIGHTS = {"values": 0.5, "subjects": 1.2, "conditions": 1.5, "group_by": 1.0, "measures": 1.0}


@dataclass(frozen=True)
class RetrieveConfig:
    max_tables: int = 6
    table_score_share: float = 0.35  # keep tables scoring at least this share of the best one
    max_terms: int = 10
    max_columns_per_table: int = 25
    min_score: float = 1.0  # below this nothing counts as a match
    max_join_hops: int = 3
    # subjects and values pick their own best table unless one already chosen is nearly as good
    subject_anchor_share: float = 0.8
    # when two tables explain a subject about equally well, keep both and let the SQL agent choose
    runner_up_share: float = 0.7
    max_tables_per_phrase: int = 2
    # other words describe the chosen tables: a chosen or linked table wins if it reaches this share of the best
    describe_share: float = 0.25
    distinct_ratio: float = 1.5  # a word adds a new table only if its best table beats the next by this much
    strong_ratio: float = 2.0  # a linked table also joins when it fits a word this much better than a chosen one


def fold(text: str) -> str:
    """Case- and accent-insensitive form: ZÜRICH, Zurich and zurich all become zurich."""
    text = unicodedata.normalize("NFKD", text.replace("ı", "i"))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", fold(text)) if t not in _STOP]


def _match_strength(q: str, d: str) -> float:
    """1 for the same word, less for an abbreviation (cust / customer), less again for the same
    stem with another ending (manager / management); 0 for different words. An exact
    match therefore outranks a looser one."""
    if q == d:
        return 1.0
    shorter = min(len(q), len(d))
    if shorter >= 4 and (q.startswith(d) or d.startswith(q)):
        return 0.8
    common = len(os.path.commonprefix([q, d]))
    return 0.6 if common >= 6 and common >= 0.7 * shorter else 0.0


def _token_match(q: str, d: str) -> bool:
    return _match_strength(q, d) > 0


class _Bm25:
    """BM25 over short documents, where a query word also matches its abbreviation."""

    def __init__(self, docs: dict[Any, list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs = {key: Counter(words) for key, words in docs.items()}
        self.lengths = {key: len(words) for key, words in docs.items()}
        self.avg = (sum(self.lengths.values()) / len(docs)) if docs else 1.0
        df = Counter(w for words in self.docs.values() for w in words)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.vocab = sorted(df)
        self.k1, self.b = k1, b

    def score(self, query: list[str]) -> dict[Any, float]:
        out: dict[Any, float] = {}
        for q in set(query):
            matches = [(w, m) for w in self.vocab if (m := _match_strength(q, w))]
            for key, counts in self.docs.items():
                best = 0.0
                for w, strength in matches:
                    tf = counts.get(w, 0)
                    if tf:
                        norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[key] / self.avg))
                        best = max(best, strength * self.idf[w] * norm)
                if best:
                    out[key] = out.get(key, 0.0) + best
        return out


class LayerIndex:
    """Search structures built once per published layer version."""

    def __init__(self, layer: dict[str, Any], profiles: dict[str, dict]):
        self.layer = layer
        self.version = layer["version"]
        self.tables = layer["tables"]
        self.terms = {t["id"]: t for t in layer["terms"]}

        def name_words(name: str) -> str:
            return name.split(".")[-1].replace("_", " ")

        self.table_index = _Bm25(
            {k: tokens(f"{name_words(k)} {name_words(k)} {t['description']} {t['row_meaning']}") for k, t in self.tables.items()}
        )
        self.column_index = _Bm25(
            {
                (k, c["name"]): tokens(f"{name_words(c['name'])} {name_words(c['name'])} {c['description']}")
                for k, t in self.tables.items()
                for c in t["columns"]
            }
        )
        self.term_index = _Bm25(
            {
                i: tokens(" ".join([t["term"], t["term"], *t["synonyms"], t["definition"]]))
                for i, t in self.terms.items()
            }
        )
        # every value seen in the data (profiles keep complete lists for short columns, top values otherwise)
        self.values: dict[str, list[tuple[str, str, Any]]] = {}
        for key, profile in profiles.items():
            if key not in self.tables:
                continue
            for c in profile["columns"]:
                if c["sensitive"]:
                    continue
                for v in c["values"]:
                    if isinstance(v["value"], str):
                        self.values.setdefault(fold(v["value"]).strip(), []).append((key, c["name"], v["value"]))

        self.graph: dict[str, list[dict]] = {}
        for link in layer["links"]:
            if not link["best"]:
                continue
            a, b = link["child"]["table"], link["parent"]["table"]
            self.graph.setdefault(a, []).append(link)
            self.graph.setdefault(b, []).append(link)

    def find_values(self, phrase: str) -> list[dict[str, Any]]:
        """Stored values equal to the phrase, ignoring case and accents."""
        hits = self.values.get(fold(phrase).strip(), [])
        return [{"phrase": phrase, "table": t, "column": c, "value": v} for t, c, v in sorted(hits, key=str)]

    def join_path(self, starts: list[str], goal: str, max_hops: int) -> list[str] | None:
        """Shortest chain of tables from any of `starts` to `goal` over verified links.

        Breadth-first; among equally short routes the strongest links (most values
        matching on both sides) win, then names, so the route never varies.
        """
        frontier, seen = [(s, [s]) for s in sorted(starts)], set(starts)
        while frontier:
            nxt = []
            for table, path in frontier:
                if table == goal:
                    return path
                if len(path) > max_hops:
                    continue
                for link in sorted(
                    self.graph.get(table, []),
                    key=lambda l: (-l["containment"] * l["parent_coverage"], l["child"]["table"], l["child"]["column"]),
                ):
                    other = link["parent"]["table"] if link["child"]["table"] == table else link["child"]["table"]
                    if other not in seen:
                        seen.add(other)
                        nxt.append((other, path + [other]))
            frontier = nxt
        return None

    def links_between(self, a: str, b: str) -> list[dict]:
        """Every verified link between two tables: one pair can be joined in several roles."""
        return sorted(
            (l for l in self.graph.get(a, []) if {l["child"]["table"], l["parent"]["table"]} == {a, b}),
            key=lambda l: (l["child"]["table"], l["child"]["column"], l["parent"]["column"]),
        )


@lru_cache(maxsize=2)
def _index_for(root: str, version: str) -> LayerIndex:
    storage = Storage(root)
    layer = storage.read(LAYERS, version)
    profiles = {k: storage.read(PROFILES, k) for k in storage.keys(PROFILES)}
    return LayerIndex(layer, profiles)


def load_index(storage: Storage) -> LayerIndex:
    current = storage.read(LAYERS, "current")
    if not current:
        raise RuntimeError("No published semantic layer; run the learn stages first")
    return _index_for(str(storage.root), current["version"])


def _column_brief(column: dict) -> dict:
    brief = {k: column[k] for k in ("name", "type", "description") if column.get(k)}
    for k in ("values", "meanings", "sensitive"):
        if column.get(k):
            brief[k] = column[k]
    return brief


def _place(
    index: LayerIndex, slot: str, ranked: list[tuple[float, str]], chosen: set[str], cfg: RetrieveConfig
) -> list[str]:
    """Which tables a phrase belongs to, given the tables already chosen.

    Subjects and values say where the data lives, so they take their own best
    table, plus a close runner-up. Other words (conditions, groupings,
    measures) describe those tables: they prefer a chosen table, then one
    linked to it, and add a new table only when they clearly point at one; a
    word that matches many tables about equally ("active", "count") picks
    nothing.
    """
    top, best = ranked[0]
    if slot in ("subjects", "values") or not chosen:
        for s, t in ranked:
            if t in chosen and s >= cfg.subject_anchor_share * top:
                return [t]
        return [t for s, t in ranked if s >= cfg.runner_up_share * top][: cfg.max_tables_per_phrase]
    neighbours = {
        other
        for t in chosen
        for l in index.graph.get(t, [])
        for other in (l["child"]["table"], l["parent"]["table"])
    }
    for s, t in ranked:
        if t in chosen and s >= cfg.describe_share * top:
            # the word fits a chosen table, but a linked one may fit it far better
            if best != t and best in neighbours and top >= cfg.strong_ratio * s:
                return [t, best]
            return [t]
    for s, t in ranked:
        if t in neighbours and s >= cfg.describe_share * top:
            return [t]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    if slot != "measures" and top >= cfg.distinct_ratio * second:
        return [best]
    return []


def retrieve(index: LayerIndex, extracted: dict[str, Any], cfg: RetrieveConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RetrieveConfig()
    term_scores: Counter = Counter()
    column_scores: Counter = Counter()
    totals: Counter = Counter()
    value_hits, unmatched, explained = [], [], []

    for slot, weight in _SLOT_WEIGHTS.items():
        # the most specific phrase first: "order lines" before "orders"
        for phrase in sorted(extracted.get(slot, []), key=lambda p: (-len(tokens(p)), p)):
            query = tokens(phrase)
            # how well each table explains this one phrase: its name, its best term, its best column
            contrib: Counter = Counter()
            for key, s in index.table_index.score(query).items():
                contrib[key] += weight * s
            best_term: Counter = Counter()
            for key, s in index.term_index.score(query).items():
                term_scores[key] += weight * s
                table = index.terms[key]["table"]
                best_term[table] = max(best_term[table], weight * s)
            best_column: Counter = Counter()
            for (table, column), s in index.column_index.score(query).items():
                column_scores[(table, column)] += weight * s
                best_column[table] = max(best_column[table], weight * s)
            contrib.update(best_term)
            contrib.update(best_column)
            hits = index.find_values(phrase)
            # an exact stored value is strong evidence for a name or a code, weak for a common word
            bonus = 3.0 if slot == "values" else 0.75
            for table in {h["table"] for h in hits}:
                contrib[table] += bonus
            value_hits += [{**h, "slot": slot} for h in hits]

            ranked = sorted(((s, t) for t, s in contrib.items() if s >= cfg.min_score), key=lambda x: (-x[0], x[1]))
            if not ranked:
                if slot in ("subjects", "conditions", "values"):
                    unmatched.append({"slot": slot, "phrase": phrase})
                continue
            totals.update({t: s for t, s in contrib.items() if s >= cfg.min_score})
            # a generic word that does not point at any one table places nothing
            for table in _place(index, slot, ranked, {e["table"] for e in explained}, cfg):
                explained.append({"slot": slot, "phrase": phrase, "table": table})

    subjects = extracted.get("subjects", [])
    lost_subjects = [u["phrase"] for u in unmatched if u["slot"] == "subjects"]
    if not explained or (subjects and len(lost_subjects) == len(subjects)):
        missing = ", ".join(f"'{u['phrase']}'" for u in unmatched) or "the question's words"
        return {
            "layer_version": index.version,
            "decision": "cannot_answer",
            "reason": f"nothing in the data matches {missing}",
            "unmatched": unmatched,
            "tables": [],
            "terms": [],
            "values": [],
            "joins": [],
            "related_tables": [],
        }

    # the tables that best explain some phrase, strongest overall first
    chosen = sorted({e["table"] for e in explained}, key=lambda t: (-totals[t], t))[: cfg.max_tables]

    # connect each table to those already chosen along the shortest verified route
    tables, unreachable = [chosen[0]], []
    for table in chosen[1:]:
        path = index.join_path(tables, table, cfg.max_join_hops)
        if path is None:
            unreachable.append(table)
            continue
        tables += [t for t in path if t not in tables]
    joins = []
    for i, a in enumerate(tables):
        for b in tables[i + 1:]:
            joins += index.links_between(a, b)

    terms = sorted(
        (i for i, s in term_scores.items() if index.terms[i]["table"] in tables and s >= cfg.min_score),
        key=lambda i: (-term_scores[i], i),
    )[: cfg.max_terms]
    used_columns = {
        (index.terms[i]["table"], c["name"])
        for i in terms
        for c in index.tables[index.terms[i]["table"]]["columns"]
        if re.search(rf"\b{re.escape(c['name'])}\b", index.terms[i]["sql"])
    }
    join_columns = {(l[side]["table"], l[side]["column"]) for l in joins for side in ("child", "parent")}
    value_columns = {(h["table"], h["column"]) for h in value_hits}

    packed_tables = []
    for key in tables:
        t = index.tables[key]
        wanted = (
            {c for (k, c), s in column_scores.items() if k == key and s >= cfg.min_score}
            | {c for k, c in used_columns | join_columns | value_columns if k == key}
            | set((t["grain"] or {}).get("columns", []))
        )
        ranked_columns = sorted(
            (c for c in t["columns"] if c["name"] in wanted),
            key=lambda c: (-column_scores.get((key, c["name"]), 0.0), c["name"]),
        )[: cfg.max_columns_per_table]
        packed_tables.append(
            {
                "table": key,
                "description": t["description"],
                "row_meaning": t["row_meaning"],
                "kind": t["kind"],
                "grain": (t["grain"] or {}).get("text"),
                "rule": t["rule"],
                "current_sql": t["current_sql"],
                "as_of_sql": t["as_of_sql"],
                "columns": [_column_brief(c) for c in ranked_columns],
                "other_columns": sorted(c["name"] for c in t["columns"] if c["name"] not in wanted),
            }
        )

    return {
        "layer_version": index.version,
        "decision": "answer",
        "tables": packed_tables,
        "terms": [
            {
                "id": i,
                "table": index.terms[i]["table"],
                "term": index.terms[i]["term"],
                "kind": index.terms[i]["kind"],
                "definition": index.terms[i]["definition"],
                "sql": index.terms[i]["sql"],
                "scope": index.terms[i]["verification"]["scope"],
                "proof": index.terms[i]["verification"]["proof"],
            }
            for i in terms
        ],
        "values": sorted(
            [h for h in value_hits if h["table"] in tables], key=lambda h: (h["phrase"], h["table"], h["column"])
        ),
        "joins": [
            {
                "from": f"{l['child']['table']}.{l['child']['column']}",
                "to": f"{l['parent']['table']}.{l['parent']['column']}",
                "cardinality": l["cardinality"],
                "notes": l["notes"],
            }
            for l in joins
        ],
        "unmatched": unmatched,
        "unreachable_tables": unreachable,
        "explained_by": explained,
        # one step away along verified links, in case the search missed a table the question needs
        "related_tables": sorted(
            {
                (other, index.tables[other]["kind"], index.tables[other]["description"][:160])
                for t in tables
                for l in index.graph.get(t, [])
                for other in (l["child"]["table"], l["parent"]["table"])
                if other not in tables
            }
        ),
    }
