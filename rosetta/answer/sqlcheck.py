"""Check a generated query before it runs.

Only a single read (SELECT or WITH) passes. Every table must exist in the
semantic layer, every column must exist in the table it is read from, and a
quoted value compared with a column must be one the column really holds.
Nothing here touches the database.
"""

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

_WRITES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command)


def check_query(sql: str, tables: dict[str, dict[str, Any]], dialect: str) -> list[str]:
    """Problems with a query; empty when it is a safe read of things that exist.

    `tables` maps "SCHEMA.TABLE" to its profile (columns with their values).
    """
    text = sql.strip().rstrip(";").strip()
    if ";" in text:
        return ["more than one statement"]
    try:
        statements = sqlglot.parse(text, read=dialect)
    except sqlglot.errors.ParseError as exc:
        return [f"does not parse: {str(exc).splitlines()[0]}"]
    if len(statements) != 1 or statements[0] is None:
        return ["more than one statement"]
    tree = statements[0]
    if any(tree.find_all(*_WRITES)) or not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return ["only a single SELECT is allowed"]

    by_name: dict[str, str] = {}
    for key in tables:
        schema, table = key.split(".", 1)
        by_name[table.upper()] = key
        by_name[key.upper()] = key

    issues: list[str] = []
    resolved: dict[int, dict] = {}
    checked: set[int] = set()
    for scope in traverse_scope(tree):  # innermost first; an outer scope repeats inner columns
        own: dict[str, dict[str, dict]] = {}  # alias -> column name -> profile
        derived = False
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                name = f"{source.db}.{source.name}".upper() if source.db else source.name.upper()
                if name not in by_name:
                    if source.name.upper() not in {c.upper() for c in scope.cte_sources}:
                        issues.append(f"table {name} does not exist")
                    else:
                        derived = True
                    continue
                own[alias.upper()] = {c["name"].upper(): c for c in tables[by_name[name]]["columns"]}
            else:
                derived = True  # a subquery or CTE: its output columns are not checked here
        # ORDER BY may name an output alias ("... AS n ORDER BY n")
        outputs = (
            {e.alias.upper() for e in scope.expression.expressions if isinstance(e, exp.Alias)}
            if isinstance(scope.expression, exp.Select)
            else set()
        )
        for col in scope.columns:
            if id(col) in checked:
                continue
            checked.add(id(col))
            name, qualifier = col.name.upper(), col.table.upper()
            if qualifier:
                if qualifier in own:
                    profile = own[qualifier].get(name)
                    if profile is None:
                        issues.append(f"column {col.table}.{col.name} does not exist")
                    else:
                        resolved[id(col)] = profile
                continue
            matches = [cols[name] for cols in own.values() if name in cols]
            if matches:
                resolved[id(col)] = matches[0]
            elif not derived and name not in outputs:
                issues.append(f"column {col.name} does not exist in the tables it reads")

    def seen(column: exp.Expression, literal: exp.Expression) -> None:
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
            return
        profile = resolved.get(id(column))
        if not profile or not profile.get("values_complete") or profile.get("sensitive"):
            return
        if literal.this not in {str(v["value"]) for v in profile["values"]}:
            issues.append(f"{column.name} never holds {literal.this!r}")

    for node in tree.find_all(exp.EQ, exp.NEQ):
        seen(node.left, node.right)
        seen(node.right, node.left)
    for node in tree.find_all(exp.In):
        for item in node.expressions:
            seen(node.this, item)
    return sorted(set(issues))
