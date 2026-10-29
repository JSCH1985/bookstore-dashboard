import ast
import copy
import math
import operator
import re

import pandas as pd

DASHBOARD_TABLES = ("orders", "users", "customers", "books")
VIEWS = ("overview",) + DASHBOARD_TABLES
QUERY_TABLES = DASHBOARD_TABLES + ("authors", "author_sets", "daily_revenue")
OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "contains", "not_contains",
       "startswith", "endswith", "regex", "is_null", "not_null")
TEXT_OPS = {"contains", "not_contains", "startswith", "endswith", "regex"}
AGG_FUNCS = ("count", "count_distinct", "sum", "mean", "median", "min", "max")
MAX_ROWS = 100
NAME_TITLES = {"Mrs", "Miss", "Prof", "Sir", "PhD", "Esq", "The", "Rep", "Sen", "Gov", "Amb", "Pres", "Msgr", "Rev", "Hon"}

TABLE_NOTES = {
    "orders": "one row per order; paid_price = quantity * unit_price in USD (EUR converted at 1.2); date is YYYY-MM-DD",
    "users": "raw user records from users.csv; several records can belong to one real person (person_id)",
    "customers": "real unique users after reconciling aliases; user_ids lists every user record of that person",
    "books": "one row per book; authors is the author set joined with ' & '",
    "authors": "one row per individual author; copies_sold counts every book they (co-)wrote",
    "author_sets": "one row per unique set of authors",
    "daily_revenue": "one row per calendar day between the first and last order (days without orders have 0)",
}


class ToolError(Exception):
    pass


def default_state(store):
    return {
        "store": store,
        "view": "overview",
        "filters": {t: [] for t in DASHBOARD_TABLES},
        "sort": {t: None for t in DASHBOARD_TABLES},
        "chart": {"start": None, "end": None, "granularity": "day"},
    }


def sanitize_state(state, store_names):
    clean = default_state(store_names[0] if store_names else None)
    if not isinstance(state, dict):
        return clean
    if state.get("store") in store_names:
        clean["store"] = state["store"]
    if state.get("view") in VIEWS:
        clean["view"] = state["view"]
    for t in DASHBOARD_TABLES:
        filters = (state.get("filters") or {}).get(t)
        if isinstance(filters, list):
            clean["filters"][t] = [f for f in filters if isinstance(f, dict)]
        sort = (state.get("sort") or {}).get(t)
        if isinstance(sort, dict) and sort.get("column"):
            clean["sort"][t] = {"column": str(sort["column"]), "descending": bool(sort.get("descending"))}
    chart = state.get("chart") or {}
    clean["chart"].update({k: chart.get(k) for k in ("start", "end") if chart.get(k)})
    if chart.get("granularity") in ("day", "week", "month"):
        clean["chart"]["granularity"] = chart["granularity"]
    return clean


class ToolContext:
    def __init__(self, registry, state):
        self.registry = registry
        self.state = state

    @property
    def store(self):
        return self.registry.get(self.state["store"])

    def table(self, name):
        if name not in QUERY_TABLES:
            raise ToolError(f"Unknown table '{name}'. Tables: {', '.join(QUERY_TABLES)}")
        return self.store.table(name)


def json_value(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 4)
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        return json_value(v.item())
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def records(df):
    return [{c: json_value(v) for c, v in zip(df.columns, row)} for row in df.itertuples(index=False, name=None)]


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _values(cond):
    values = cond.get("values")
    if values in (None, [], ""):
        value = cond.get("value")
        values = [] if value is None or value == "" else value if isinstance(value, list) else [value]
    return list(values)


def _number(value, column):
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        raise ToolError(f"Column '{column}' is numeric; '{value}' is not a number")


def condition_mask(df, cond):
    column, op = cond.get("column"), cond.get("op")
    if column not in df.columns:
        raise ToolError(f"Unknown column '{column}'. Columns: {', '.join(df.columns)}")
    if op not in OPS:
        raise ToolError(f"Unknown op '{op}'. Ops: {', '.join(OPS)}")
    s = df[column]
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return s.notna()
    values = _values(cond)
    if not values:
        raise ToolError(f"Op '{op}' needs 'value' or 'values'")
    case = bool(cond.get("case_sensitive", False))
    numeric = _is_numeric(s)

    if op in TEXT_OPS:
        text = pd.Series([("" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v)) for v in s], index=s.index)
        if op == "regex":
            pattern = "|".join(f"(?:{v})" for v in values)
            if len(pattern) > 500:
                raise ToolError("Regex too long")
            try:
                compiled = re.compile(pattern, 0 if case else re.IGNORECASE)
            except re.error as e:
                raise ToolError(f"Invalid regex: {e}")
            mask = text.map(lambda t: bool(compiled.search(t)))
        else:
            hay = text if case else text.str.casefold()
            needles = [str(v) if case else str(v).casefold() for v in values]
            if op == "startswith":
                mask = hay.map(lambda t: t.startswith(tuple(needles)))
            elif op == "endswith":
                mask = hay.map(lambda t: t.endswith(tuple(needles)))
            else:
                mask = hay.map(lambda t: any(n in t for n in needles))
                if op == "not_contains":
                    mask = ~mask
        return mask.astype(bool)

    if op in ("eq", "ne", "in", "not_in"):
        if numeric:
            mask = s.isin([_number(v, column) for v in values])
        else:
            keys = s.map(lambda v: None if v is None or v != v else (str(v) if case else str(v).casefold()))
            mask = keys.isin([str(v) if case else str(v).casefold() for v in values])
        mask = mask.fillna(False).astype(bool)
        return ~mask if op in ("ne", "not_in") else mask

    if op == "between" and len(values) != 2:
        raise ToolError("Op 'between' needs values [low, high]")
    present = s.notna()
    if numeric:
        values = [_number(v, column) for v in values]
        base = s.astype(float)
    else:
        values = [str(v) for v in values]
        base = s.map(lambda v: str(v) if v is not None and v == v else "")
    compare = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}
    if op == "between":
        mask = (base >= values[0]) & (base <= values[1])
    else:
        mask = compare[op](base, values[0])
    return (mask & present).fillna(False).astype(bool)


def build_mask(df, conditions, match="all"):
    if not conditions:
        raise ToolError("At least one condition is required")
    if match not in ("all", "any"):
        raise ToolError("match must be 'all' or 'any'")
    masks = [condition_mask(df, c) for c in conditions]
    result = masks[0]
    for m in masks[1:]:
        result = (result & m) if match == "all" else (result | m)
    return result


def apply_filters(df, filters):
    errors = []
    for f in filters:
        try:
            mask = build_mask(df, f.get("conditions") or [], f.get("match", "all"))
        except ToolError as e:
            errors.append(f"{f.get('label', 'filter')}: {e}")
            continue
        df = df[~mask] if f.get("mode") == "exclude" else df[mask]
    return df, errors


def describe_condition(cond):
    values = _values(cond)
    shown = ", ".join(f'"{v}"' for v in values[:3]) + (f" (+{len(values) - 3} more)" if len(values) > 3 else "")
    words = {"eq": "=", "ne": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "in": "is one of",
             "not_in": "is not one of", "contains": "contains", "not_contains": "does not contain",
             "startswith": "starts with", "endswith": "ends with", "regex": "matches", "between": "between"}
    if cond["op"] == "is_null":
        return f"{cond['column']} is empty"
    if cond["op"] == "not_null":
        return f"{cond['column']} is not empty"
    return f"{cond['column']} {words[cond['op']]} {shown}"


def _normalize_conditions(conditions):
    if not isinstance(conditions, list):
        raise ToolError("conditions must be a list")
    out = []
    for c in conditions:
        if not isinstance(c, dict):
            raise ToolError("each condition must be an object")
        item = {"column": c.get("column"), "op": c.get("op")}
        values = _values(c)
        if values:
            item["values"] = [json_value(v) for v in values]
        if c.get("case_sensitive"):
            item["case_sensitive"] = True
        out.append(item)
    return out


def _filtered(ctx, table, filters, match, use_dashboard):
    df = ctx.table(table)
    if use_dashboard and table in DASHBOARD_TABLES:
        df, _ = apply_filters(df, ctx.state["filters"][table])
    if filters:
        df = df[build_mask(df, _normalize_conditions(filters), match)]
    return df


def get_dashboard_metrics(ctx):
    return ctx.store.metrics


def describe_data(ctx):
    store = ctx.store
    tables = {}
    for name in QUERY_TABLES:
        df = store.table(name)
        cols = {c: ("number" if _is_numeric(df[c]) else "text") for c in df.columns}
        tables[name] = {"rows": len(df), "note": TABLE_NOTES[name], "columns": cols}
    return {"store": store.name, "tables": tables, "data_quality": store.quality}


def query_table(ctx, table, filters=None, match="all", group_by=None, aggregations=None, columns=None,
                sort_by=None, descending=None, limit=20, use_dashboard_filters=False):
    df = _filtered(ctx, table, filters, match, use_dashboard_filters)
    matched = len(df)
    limit = max(1, min(int(limit or 20), MAX_ROWS))
    group_by = [group_by] if isinstance(group_by, str) else list(group_by or [])
    aggregations = list(aggregations or [])
    for g in group_by:
        if g not in df.columns:
            raise ToolError(f"Unknown group_by column '{g}'. Columns: {', '.join(df.columns)}")

    if group_by or aggregations:
        if not aggregations:
            aggregations = [{"func": "count", "column": "*"}]
        specs = []
        for a in aggregations:
            func, col = a.get("func"), a.get("column") or "*"
            if func not in AGG_FUNCS:
                raise ToolError(f"Unknown func '{func}'. Funcs: {', '.join(AGG_FUNCS)}")
            if col == "*" and func != "count":
                raise ToolError("Only 'count' can use column '*'")
            if col != "*" and col not in df.columns:
                raise ToolError(f"Unknown column '{col}'. Columns: {', '.join(df.columns)}")
            if func in ("sum", "mean", "median") and not _is_numeric(df[col]):
                raise ToolError(f"Column '{col}' is not numeric")
            specs.append(("rows" if col == "*" else f"{func}_{col}", col, func))
        if group_by:
            grouped = df.groupby(group_by, dropna=False, sort=False)
            pieces = {}
            for name, col, func in specs:
                if col == "*":
                    pieces[name] = grouped.size()
                elif func == "count_distinct":
                    pieces[name] = grouped[col].nunique()
                else:
                    pieces[name] = getattr(grouped[col], func)()
            out = pd.DataFrame(pieces).reset_index()
            if sort_by is None:
                sort_by, descending = specs[0][0], True if descending is None else descending
        else:
            row = {}
            for name, col, func in specs:
                if col == "*":
                    row[name] = len(df)
                elif func == "count_distinct":
                    row[name] = df[col].nunique()
                else:
                    values = df[col].dropna()
                    row[name] = getattr(values, func)() if len(values) else None
            out = pd.DataFrame([row])
    else:
        out = df
        if columns:
            missing = [c for c in columns if c not in out.columns]
            if missing:
                raise ToolError(f"Unknown columns {missing}. Columns: {', '.join(out.columns)}")
            out = out[list(columns)]

    if sort_by:
        if sort_by not in out.columns:
            raise ToolError(f"Cannot sort by '{sort_by}'. Columns: {', '.join(out.columns)}")
        out = out.sort_values(sort_by, ascending=not descending, kind="stable", na_position="last")
    shown = out.head(limit).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].round(2)
    return {
        "table": table, "matched_rows": matched, "result_rows": len(out),
        "returned_rows": len(shown), "truncated": len(out) > limit, "rows": records(shown),
    }


def distinct_values(ctx, table, column, filters=None, match="all", limit=1000, use_dashboard_filters=False):
    df = _filtered(ctx, table, filters, match, use_dashboard_filters)
    if column not in df.columns:
        raise ToolError(f"Unknown column '{column}'. Columns: {', '.join(df.columns)}")
    counts = df[column].value_counts(dropna=True)
    counts = counts.rename_axis("value").reset_index(name="rows").sort_values(["rows", "value"], ascending=[False, True])
    limit = max(1, min(int(limit or 1000), 2000))
    return {"table": table, "column": column, "distinct_count": len(counts),
            "truncated": len(counts) > limit, "values": records(counts.head(limit))}


def find_customer(ctx, text, limit=10):
    text = str(text).strip()
    if not text:
        raise ToolError("text is required")
    users = ctx.store.table("users")
    hay = users[["name", "email", "address", "phone"]].astype(str).apply(lambda col: col.str.casefold())
    mask = hay.apply(lambda col: col.str.contains(text.casefold(), regex=False)).any(axis=1)
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 4:
        mask |= users["phone"].astype(str).map(lambda p: digits in re.sub(r"\D", "", p))
    person_ids = list(dict.fromkeys(users.loc[mask, "person_id"]))[: max(1, min(int(limit or 10), 50))]
    customers = ctx.store.table("customers")
    rows = customers[customers["person_id"].isin(person_ids)]
    return {"matches": len(set(users.loc[mask, "person_id"])), "customers": records(rows),
            "user_records": records(users[users["person_id"].isin(person_ids)])}


def _name_lexicon(store):
    names = list(store.table("users")["name"].dropna()) + list(store.table("authors")["author"].dropna())
    lexicon = set()
    for name in names:
        for token in re.findall(r"[^\W\d_]+", name):
            if len(token) >= 3 and token[0].isupper() and not token.isupper() and token not in NAME_TITLES:
                lexicon.add(token)
    return lexicon


def find_name_candidates(ctx, table, column, limit=500):
    df = ctx.table(table)
    if column not in df.columns:
        raise ToolError(f"Unknown column '{column}'. Columns: {', '.join(df.columns)}")
    lexicon = _name_lexicon(ctx.store)
    found = []
    for value, count in df[column].dropna().astype(str).value_counts().items():
        tokens = {t for t in re.findall(r"[^\W\d_]+", value) if t[0].isupper()}
        hits = sorted(t for t in tokens if t in lexicon)
        if hits:
            found.append({"value": value, "rows": int(count), "known_name_tokens": hits})
    found.sort(key=lambda r: r["value"])
    return {"table": table, "column": column, "candidates": found[: int(limit or 500)], "candidate_count": len(found),
            "note": "Tokens matching first/last names seen in users and authors. Common words that are also names "
                    "can be false positives and names absent from the data are missed; review with distinct_values."}


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": lambda *a: sum(a)}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ToolError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError("Only numbers, + - * / % ** // and round/abs/min/max/sum are allowed")


def calculate(ctx, expression):
    expression = str(expression).replace(",", "") if re.search(r"\d,\d{3}", str(expression)) else str(expression)
    try:
        result = _eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolError(f"Cannot evaluate: {e}")
    return {"expression": expression, "result": round(result, 6) if isinstance(result, float) else result}


def _check_table(table):
    if table not in DASHBOARD_TABLES:
        raise ToolError(f"Dashboard tables: {', '.join(DASHBOARD_TABLES)}")


def set_filter(ctx, table, conditions, match="all", mode="keep", replace_existing=False):
    _check_table(table)
    if mode not in ("keep", "exclude"):
        raise ToolError("mode must be 'keep' or 'exclude'")
    conditions = _normalize_conditions(conditions)
    full = ctx.table(table)
    matching = build_mask(full, conditions, match)
    current = [] if replace_existing else ctx.state["filters"][table]
    before, _ = apply_filters(full, current)
    label = (" AND " if match == "all" else " OR ").join(describe_condition(c) for c in conditions)
    entry = {"mode": mode, "match": match, "conditions": conditions,
             "label": ("only " if mode == "keep" else "hide ") + label}
    filters = current + [entry]
    after, _ = apply_filters(full, filters)
    ctx.state["filters"][table] = filters
    ctx.state["view"] = table
    return {"table": table, "applied": entry["label"], "rows_in_table": len(full),
            "rows_matching_condition": int(matching.sum()), "rows_shown_before": len(before),
            "rows_shown_now": len(after), "active_filters": [f.get("label") for f in filters],
            "first_rows_shown": records(after.head(5))}


def clear_filters(ctx, table="all"):
    tables = DASHBOARD_TABLES if table in (None, "", "all") else [table]
    for t in tables:
        _check_table(t)
        ctx.state["filters"][t] = []
    return {"cleared": list(tables), "rows_shown_now": {t: len(ctx.table(t)) for t in tables}}


def sort_table(ctx, table, column, descending=False):
    _check_table(table)
    if column not in ctx.table(table).columns:
        raise ToolError(f"Unknown column '{column}'. Columns: {', '.join(ctx.table(table).columns)}")
    ctx.state["sort"][table] = {"column": column, "descending": bool(descending)}
    ctx.state["view"] = table
    return {"table": table, "sorted_by": column, "descending": bool(descending)}


def show_view(ctx, view):
    if view not in VIEWS:
        raise ToolError(f"Views: {', '.join(VIEWS)}")
    ctx.state["view"] = view
    return {"view": view}


def _date_arg(value, name):
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")
    except ValueError:
        raise ToolError(f"{name} must be a date like 2025-09-01")


def set_chart(ctx, start_date=None, end_date=None, granularity="day"):
    if granularity not in ("day", "week", "month"):
        raise ToolError("granularity must be day, week or month")
    start, end = _date_arg(start_date, "start_date"), _date_arg(end_date, "end_date")
    daily = ctx.table("daily_revenue")
    window = daily[(daily["date"] >= (start or "0000")) & (daily["date"] <= (end or "9999"))]
    ctx.state["chart"] = {"start": start, "end": end, "granularity": granularity}
    ctx.state["view"] = "overview"
    return {"start": start or daily["date"].min(), "end": end or daily["date"].max(), "granularity": granularity,
            "days_in_range": len(window), "revenue_in_range": round(float(window["revenue"].sum()), 2)}


def set_store(ctx, store):
    names = ctx.registry.names()
    if store not in names:
        raise ToolError(f"Stores: {', '.join(names)}")
    ctx.state.clear()
    ctx.state.update(default_state(store))
    return {"store": store, "metrics": ctx.store.metrics}


COND_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string"},
        "op": {"type": "string", "enum": list(OPS)},
        "value": {"type": "string", "description": "Single value. Numbers and dates (YYYY-MM-DD) as text."},
        "values": {"type": "array", "items": {"type": "string"},
                   "description": "Several values: for in/not_in, [low, high] for between, or any-of for contains/startswith/endswith/regex."},
        "case_sensitive": {"type": "boolean", "description": "Text comparisons ignore case unless true."},
    },
    "required": ["column", "op"],
}
FILTERS_SCHEMA = {"type": "array", "items": COND_SCHEMA, "description": "Row conditions."}
MATCH_SCHEMA = {"type": "string", "enum": ["all", "any"], "description": "Combine conditions with AND (all) or OR (any)."}


def _tool(fn, description, properties=None, required=None):
    function = {"name": fn.__name__, "description": description}
    if properties:
        function["parameters"] = {"type": "object", "properties": properties, "required": required or []}
    return {"fn": fn, "schema": {"type": "function", "function": function}}


TOOLS = {t["schema"]["function"]["name"]: t for t in [
    _tool(get_dashboard_metrics,
          "Exact headline results shown on the dashboard: unique (real) users, unique author sets, most popular "
          "author set(s) and individual author(s) by copies sold, best buyer(s) with all user_ids and total spent, "
          "top 5 revenue days, total revenue, order count, date range, and the automatic data checks "
          "(dropped rows, orders dated on/after the processing day, one-day revenue spikes)."),
    _tool(describe_data, "Tables, columns, row counts and the data-cleaning report of the current store."),
    _tool(query_table,
          "Exact query over one table: filter rows, then either list rows or group and aggregate. All counting, "
          "summing and averaging must be done with this tool, never mentally.",
          {"table": {"type": "string", "enum": list(QUERY_TABLES)},
           "filters": FILTERS_SCHEMA, "match": MATCH_SCHEMA,
           "group_by": {"type": "array", "items": {"type": "string"}},
           "aggregations": {"type": "array", "items": {"type": "object", "properties": {
               "func": {"type": "string", "enum": list(AGG_FUNCS)},
               "column": {"type": "string", "description": "Column name, or * with count for row count."}},
               "required": ["func", "column"]}},
           "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to return when listing rows."},
           "sort_by": {"type": "string", "description": "A column or an aggregation alias such as sum_paid_price."},
           "descending": {"type": "boolean"},
           "limit": {"type": "integer", "description": "Rows to return, max 100. matched_rows is always exact."},
           "use_dashboard_filters": {"type": "boolean", "description": "Start from the rows currently shown on the dashboard."}},
          ["table"]),
    _tool(distinct_values, "Distinct values of a column with row counts (e.g. all book titles or genres). "
                           "Returns up to 1000 values by default (limit max 2000); check 'truncated'.",
          {"table": {"type": "string", "enum": list(QUERY_TABLES)}, "column": {"type": "string"},
           "filters": FILTERS_SCHEMA, "match": MATCH_SCHEMA, "limit": {"type": "integer"},
           "use_dashboard_filters": {"type": "boolean"}}, ["table", "column"]),
    _tool(find_customer, "Find real customers by part of a name, e-mail, phone or address; returns every alias record.",
          {"text": {"type": "string"}, "limit": {"type": "integer"}}, ["text"]),
    _tool(find_name_candidates,
          "Values of a text column containing words that are first/last names of people in this dataset. "
          "Helper for questions like 'titles with people names'.",
          {"table": {"type": "string", "enum": list(QUERY_TABLES)}, "column": {"type": "string"}}, ["table", "column"]),
    _tool(calculate, "Exact arithmetic on numbers taken from tool results (differences, ratios, percentages).",
          {"expression": {"type": "string", "description": "e.g. (3719.35 - 3587.94) / 3587.94 * 100"}}, ["expression"]),
    _tool(set_filter,
          "DASHBOARD CONTROL. Filter a dashboard table. mode 'keep' shows only matching rows (select / show only); "
          "mode 'exclude' hides matching rows (filter out / remove / hide). Filters stack unless replace_existing.",
          {"table": {"type": "string", "enum": list(DASHBOARD_TABLES)}, "conditions": FILTERS_SCHEMA,
           "match": MATCH_SCHEMA, "mode": {"type": "string", "enum": ["keep", "exclude"]},
           "replace_existing": {"type": "boolean"}}, ["table", "conditions", "mode"]),
    _tool(clear_filters, "DASHBOARD CONTROL. Remove filters from one dashboard table or all.",
          {"table": {"type": "string", "enum": list(DASHBOARD_TABLES) + ["all"]}}),
    _tool(sort_table, "DASHBOARD CONTROL. Sort a dashboard table.",
          {"table": {"type": "string", "enum": list(DASHBOARD_TABLES)}, "column": {"type": "string"},
           "descending": {"type": "boolean"}}, ["table", "column"]),
    _tool(show_view, "DASHBOARD CONTROL. Open a dashboard tab.",
          {"view": {"type": "string", "enum": list(VIEWS)}}, ["view"]),
    _tool(set_chart, "DASHBOARD CONTROL. Zoom the revenue chart to a date range and/or change granularity. "
                     "Empty dates reset to the full range.",
          {"start_date": {"type": "string"}, "end_date": {"type": "string"},
           "granularity": {"type": "string", "enum": ["day", "week", "month"]}}),
    _tool(set_store, "DASHBOARD CONTROL. Switch the dashboard to another store folder (resets filters).",
          {"store": {"type": "string"}}, ["store"]),
]}


def schemas():
    return [t["schema"] for t in TOOLS.values()]


def execute(name, args, ctx):
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return tool["fn"](ctx, **(args or {}))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def table_page(ctx, table, page=1, page_size=25):
    _check_table(table)
    full = ctx.table(table)
    df, errors = apply_filters(full, ctx.state["filters"][table])
    sort = ctx.state["sort"][table]
    if sort and sort["column"] in df.columns:
        key = (lambda c: c.astype(str).str.casefold()) if not _is_numeric(df[sort["column"]]) else None
        df = df.sort_values(sort["column"], ascending=not sort["descending"], kind="stable", na_position="last", key=key)
    page_size = max(5, min(int(page_size or 25), 200))
    pages = max(1, math.ceil(len(df) / page_size))
    page = max(1, min(int(page or 1), pages))
    chunk = df.iloc[(page - 1) * page_size: page * page_size]
    return {"table": table, "columns": list(full.columns), "rows": records(chunk), "total": len(full),
            "filtered": len(df), "page": page, "pages": pages, "page_size": page_size,
            "filters": [f.get("label") for f in ctx.state["filters"][table]], "sort": sort, "errors": errors}


def snapshot(state):
    return copy.deepcopy(state)
