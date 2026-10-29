import json
import pickle
import re
import sys
import threading
import warnings
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from dateutil import parser as date_parser

EUR_TO_USD = Decimal("1.2")
MISSING = {"", "null", "none", "nan", "nat", "<na>", "n/a", "na", "-", "--", "nil", "undefined", "?"}
NAME_SUFFIXES = {"jr", "sr", "md", "phd", "dds", "dvm", "esq", "i", "ii", "iii", "iv", "v"}
COLUMN_ALIASES = {
    "e_mail": "email", "mail": "email", "phone_number": "phone", "telephone": "phone",
    "full_name": "name", "authors": "author", "qty": "quantity", "price": "unit_price",
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
DATE_RE = re.compile(r"(?<![\d:./-])(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})(?![\d:])")
AMPM_RE = re.compile(r"(?<![a-z])([ap])\.?\s?m\.?(?![a-z])", re.I)
DEFAULT_DT = datetime(1900, 1, 1)


def clean_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.casefold() in MISSING else text


def clean_text(series):
    return pd.Series([clean_value(v) for v in series], index=series.index, dtype=object)


def normalize_columns(df):
    df = df.copy()
    names = []
    for col in df.columns:
        name = re.sub(r"[\s\-]+", "_", str(col).strip().lstrip(":").strip().lower())
        names.append(COLUMN_ALIASES.get(name, name))
    df.columns = names
    return df


def to_int(series):
    num = pd.to_numeric(clean_text(series), errors="coerce")
    return num.where(num == num.round()).astype("Int64")


def parse_price(value):
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        if value != value:
            return None, None
        return Decimal(str(value)), None
    text = clean_value(value)
    if text is None:
        return None, None
    upper = text.upper()
    eur = "€" in text or "EUR" in upper
    usd = "$" in text or "USD" in upper
    if eur and usd:
        return None, None
    currency = "EUR" if eur else "USD" if usd else None
    parts = list(re.finditer(r"\d+", text))
    try:
        if len(parts) == 1:
            digits = parts[0].group()
            tail = text[parts[0].end():].lstrip()
            amount = Decimal(digits) / 100 if tail.startswith("¢") else Decimal(digits)
        elif len(parts) == 2:
            whole, frac = parts[0].group(), parts[1].group()
            separator = text[parts[0].end():parts[1].start()].strip()
            if separator in {".", ","}:
                amount = Decimal(f"{whole}.{frac}")
            elif len(frac) <= 2:
                amount = Decimal(whole) + Decimal(frac) / 100
            else:
                return None, currency
        elif len(parts) >= 3:
            last = parts[-1]
            separator = text[parts[-2].end():last.start()].strip()
            if separator not in {".", ","} or len(last.group()) > 2:
                return None, currency
            amount = Decimal("".join(p.group() for p in parts[:-1]) + "." + last.group())
        else:
            return None, currency
    except InvalidOperation:
        return None, currency
    return amount, currency


def price_to_cents(amount, currency, quantity=1):
    usd = amount * quantity * (EUR_TO_USD if currency == "EUR" else 1)
    return int((usd * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_quantity(value):
    text = clean_value(value)
    if text is None:
        return None
    if text.casefold() in NUMBER_WORDS:
        return NUMBER_WORDS[text.casefold()]
    match = re.search(r"\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    qty = float(match.group().replace(",", "."))
    return int(qty) if qty > 0 and qty == int(qty) else None


def detect_day_first(values):
    stats = {}
    for text in values:
        for m in DATE_RE.finditer(text):
            a, sep, b, c = m.groups()
            if len(a) <= 2 and len(c) in (2, 4):
                max_a, max_b = stats.get(sep, (0, 0))
                stats[sep] = (max(max_a, int(a)), max(max_b, int(b)))
    decision = {}
    for sep, (max_a, max_b) in stats.items():
        if max_a > 12 and max_b <= 12:
            decision[sep] = True
        elif max_b > 12 and max_a <= 12:
            decision[sep] = False
        elif max_a <= 12 and max_b <= 12:
            decision[sep] = sep != "/"
        else:
            decision[sep] = None
    return decision


def _to_iso_dates(text, day_first):
    def replace(m):
        a, sep, b, c = m.groups()
        if len(a) == 4:
            year, month, day = int(a), int(b), int(c)
        elif len(a) <= 2 and len(c) in (2, 4):
            first = day_first.get(sep)
            if first is None:
                first = int(a) > 12 or (int(b) <= 12 and sep != "/")
            day, month = (int(a), int(b)) if first else (int(b), int(a))
            year = int(c) + 2000 if len(c) == 2 else int(c)
        else:
            return m.group(0)
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return m.group(0)
        return f" {year:04d}-{month:02d}-{day:02d} "
    return DATE_RE.sub(replace, text)


def parse_timestamp(text, day_first):
    if re.fullmatch(r"\d{10}(\.\d+)?", text):
        return pd.Timestamp(float(text), unit="s").to_pydatetime()
    if re.fullmatch(r"\d{13}", text):
        return pd.Timestamp(int(text), unit="ms").to_pydatetime()
    norm = _to_iso_dates(text, day_first)
    norm = AMPM_RE.sub(lambda m: f" {m.group(1).upper()}M ", norm)
    norm = re.sub(r"[;,|]", " ", norm)
    norm = re.sub(r"\b(at|on)\b", " ", norm, flags=re.I)
    norm = re.sub(r"\s+", " ", norm).strip()
    result = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fuzzy in (False, True):
            try:
                result = date_parser.parse(norm, default=DEFAULT_DT, fuzzy=fuzzy)
                break
            except (ValueError, OverflowError, TypeError):
                continue
    if result is None:
        return None
    result = result.replace(tzinfo=None)
    return result if 1990 <= result.year <= 2100 else None


def parse_timestamps(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(series)
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_localize(None)
        return values, {}
    texts = clean_text(series)
    unique = [t for t in pd.unique(texts) if t is not None]
    day_first = detect_day_first(unique)
    parsed = {t: parse_timestamp(t, day_first) for t in unique}
    values = pd.to_datetime(texts.map(lambda t: parsed.get(t) if t is not None else None), errors="coerce")
    orientation = {sep: ("day first" if v else "month first" if v is False else "mixed, decided per value") for sep, v in day_first.items()}
    return values, orientation


def phone_key(value):
    text = clean_value(value)
    if text is None:
        return None, None
    ext_match = re.search(r"(?:x|ext\.?|#)\s*(\d+)\s*$", text, flags=re.I)
    ext = ext_match.group(1) if ext_match else ""
    main = text[:ext_match.start()] if ext_match else text
    digits = re.sub(r"\D", "", main)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if not digits:
        return None, None
    display = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}" if len(digits) == 10 else digits
    key = digits + (f"x{ext}" if ext else "")
    return key, display + (f" x{ext}" if ext else "")


def text_key(value):
    if value is None:
        return None
    key = re.sub(r"[^\w\s]", " ", value.casefold())
    key = re.sub(r"\s+", " ", key).strip()
    return key or None


def split_authors(value):
    if isinstance(value, (list, tuple)):
        raw = [clean_value(v) for v in value]
    else:
        text = clean_value(value)
        raw = re.split(r"\s*[,;&]\s*", text) if text else []
    names = []
    for part in raw:
        part = clean_value(part)
        if not part:
            continue
        if names and part.casefold().rstrip(".") in NAME_SUFFIXES:
            names[-1] = f"{names[-1]} {part}"
        else:
            names.append(part)
    unique = {}
    for name in names:
        unique.setdefault(name.casefold(), name)
    return tuple(unique[k] for k in sorted(unique))


class _YamlLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
    pass


def _construct_tagged(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_YamlLoader.add_multi_constructor("!", _construct_tagged)


def read_yaml_records(path):
    with open(path, encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_YamlLoader)
    if isinstance(data, dict):
        values = list(data.values())
        if len(values) == 1 and isinstance(values[0], list):
            data = values[0]
        elif values and all(isinstance(v, dict) for v in values):
            data = [{"id": k, **v} for k, v in data.items()]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported structure in {path}")
    return [r for r in data if isinstance(r, dict)]


def load_orders(path):
    raw = normalize_columns(pd.read_parquet(path)).rename(columns={"id": "order_id"})
    original = raw.astype(str)
    report = {"raw_rows": len(raw)}
    text_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_datetime64_any_dtype(raw[c])]
    for col in text_cols:
        raw[col] = clean_text(raw[col])
    before = len(raw)
    raw = raw.drop_duplicates()
    report["exact_duplicates_removed"] = before - len(raw)

    df = raw.copy()
    for col in ("order_id", "user_id", "book_id"):
        df[col] = to_int(raw[col])
    df["quantity"] = [parse_quantity(v) for v in raw["quantity"]]
    parsed = [parse_price(v) for v in raw["unit_price"]]
    df["unit_price_raw"] = raw["unit_price"].map(clean_value)
    df["price_currency"] = [cur or ("USD" if amt is not None else None) for amt, cur in parsed]
    df["price_amount"] = [amt if amt is not None and Decimal("0") < amt < Decimal("100000") else None for amt, _ in parsed]
    df["unit_price_cents"] = [price_to_cents(amt, cur) if amt is not None else None
                              for amt, cur in zip(df["price_amount"], df["price_currency"])]
    report["distinct_price_formats"] = int(raw["unit_price"].dropna().astype(str).str.replace(r"\d+", "9", regex=True).nunique())
    report["distinct_timestamp_formats"] = int(raw["timestamp"].dropna().astype(str).str.replace(r"\d+", "9", regex=True).nunique())
    report["prices_without_currency_assumed_usd"] = sum(1 for amt, cur in parsed if amt is not None and cur is None)
    report["eur_prices_converted"] = sum(1 for _, cur in parsed if cur == "EUR")
    df["timestamp"], report["slash_dot_dash_date_orientation"] = parse_timestamps(raw["timestamp"])

    invalid = {
        "order_id": df["order_id"].isna(),
        "user_id": df["user_id"].isna(),
        "book_id": df["book_id"].isna(),
        "quantity": df["quantity"].isna(),
        "unit_price": df["unit_price_cents"].isna(),
        "timestamp": df["timestamp"].isna(),
    }
    report["invalid_values"] = {
        k: {"count": int(m.sum()), "examples": list(original.loc[m[m].index, k].head(5))}
        for k, m in invalid.items() if m.any()
    }
    bad = pd.concat(invalid, axis=1).any(axis=1)
    df = df[~bad].copy()
    report["rows_dropped_unrecoverable"] = int(bad.sum())

    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    report["duplicate_order_ids_removed"] = before - len(df)

    df["quantity"] = df["quantity"].astype(int)
    df["unit_price_cents"] = df["unit_price_cents"].astype(int)
    df["paid_cents"] = [price_to_cents(a, c, q) for a, c, q in zip(df["price_amount"], df["price_currency"], df["quantity"])]
    df = df.drop(columns="price_amount")
    df["unit_price"] = df["unit_price_cents"] / 100
    df["paid_price"] = df["paid_cents"] / 100
    report["unit_price_usd_range"] = [float(df["unit_price"].min()), float(df["unit_price"].max())]
    df["date"] = df["timestamp"].dt.normalize()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    report["rows"] = len(df)
    return df.reset_index(drop=True), report


def load_users(path):
    raw = normalize_columns(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
    raw = raw.rename(columns={"id": "user_id"})
    report = {"raw_rows": len(raw)}
    for col in raw.columns:
        raw[col] = clean_text(raw[col])
    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    report["exact_duplicates_removed"] = before - len(raw)

    df = raw.copy()
    df["user_id"] = to_int(raw["user_id"])
    missing_id = df["user_id"].isna()
    report["rows_without_valid_id_dropped"] = int(missing_id.sum())
    df = df[~missing_id].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["user_id"], keep="first").reset_index(drop=True)
    report["conflicting_duplicate_ids_removed"] = before - len(df)

    df["email"] = df["email"].map(lambda v: re.sub(r"\s+", "", v).lower() if v else None)
    phones = [phone_key(v) for v in df["phone"]]
    df["phone_key"] = [k for k, _ in phones]
    df["phone"] = [d for _, d in phones]
    df["name_key"] = df["name"].map(text_key)
    df["address_key"] = df["address"].map(text_key)
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in ("name", "address", "phone", "email")}

    df["person_id"] = reconcile(df)
    sizes = df.groupby("person_id")["user_id"].transform("size")
    df["aliases"] = sizes.astype(int)
    report["rows"] = len(df)
    report["real_users"] = int(df["person_id"].nunique())
    return df, report


def reconcile(df):
    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = df[["name_key", "address_key", "phone_key", "email"]].reset_index(drop=True)
    for combo in combinations(keys.columns, 3):
        subset = keys[list(combo)].dropna()
        for positions in subset.groupby(list(combo), sort=False).indices.values():
            members = subset.index[positions]
            root = find(members[0])
            for member in members[1:]:
                other = find(member)
                if other != root:
                    parent[max(root, other)] = min(root, other)
                    root = min(root, other)
    roots = [find(i) for i in range(n)]
    ids = df["user_id"].reset_index(drop=True)
    first_id = pd.Series(ids.values).groupby(roots).transform("min")
    return pd.Series(first_id.values, index=df.index).astype(int)


def load_books(path):
    raw = normalize_columns(pd.DataFrame(read_yaml_records(path))).rename(columns={"id": "book_id"})
    report = {"raw_rows": len(raw)}
    authors = [split_authors(v) for v in raw.get("author", pd.Series([None] * len(raw)))]
    for col in raw.columns:
        if col != "author":
            raw[col] = clean_text(raw[col])
    df = raw.drop(columns=["author"], errors="ignore").copy()
    df["author_set"] = authors
    df["book_id"] = to_int(df["book_id"])
    df = df[df["book_id"].notna()].copy()
    before = len(df)
    df = df.drop_duplicates(subset=["book_id"], keep="first")
    report["duplicate_ids_removed"] = before - len(df)
    if "year" in df:
        year = pd.to_numeric(df["year"], errors="coerce")
        valid = year.between(1000, datetime.now().year + 1) & (year == year.round())
        report["missing_or_invalid_years"] = int((~valid).sum())
        df["year"] = year.where(valid).astype("Int64")
    df["authors"] = df["author_set"].map(lambda s: " & ".join(s) if s else None)
    df["author_count"] = df["author_set"].map(len)
    report["books_without_authors"] = int((df["author_count"] == 0).sum())
    report["rows"] = len(df)
    return df.reset_index(drop=True), report


def _join_unique(values):
    seen = []
    for v in values:
        if v is not None and v == v and v not in seen:
            seen.append(v)
    return " | ".join(str(v) for v in seen) if seen else None


def data_checks(daily_cents, order_dates, report, processed_on):
    revenue = daily_cents / 100
    typical = revenue[revenue > 0].median()
    month = revenue.rolling(31, center=True, min_periods=1).median()
    big = revenue > 2 * typical
    ratio = (revenue / month.clip(lower=0.01)).where(big, 0)
    spikes = [d.strftime("%Y-%m-%d") for d in revenue.index[(ratio > 3).to_numpy()]]
    late = int((order_dates >= processed_on).sum())
    dropped = report.get("rows_dropped_unrecoverable", 0)
    mixed = [sep for sep, v in report.get("slash_dot_dash_date_orientation", {}).items() if v.startswith("mixed")]
    day = processed_on.strftime("%Y-%m-%d")
    checks = [
        {"name": "Every order kept", "status": "ok" if not dropped else "warning",
         "detail": f"{report['raw_rows'] - dropped:,} of {report['raw_rows']:,} orders parsed; {dropped} dropped"},
        {"name": "Nothing dated on or after the processing day", "status": "ok" if not late else "error",
         "detail": f"{late} orders dated {day} or later" + ("; timestamps that failed to parse often land on today" if late else "")},
        {"name": "No one-day revenue spikes", "status": "ok" if not spikes else "error",
         "detail": f"busiest day is {ratio.max():.1f}x its surrounding month (alarm above 3x)" if not spikes
                   else "suspicious spike on " + ", ".join(spikes)},
        {"name": "Day/month order unambiguous", "status": "ok" if not mixed else "warning",
         "detail": "decided from the data for each date separator" if not mixed else "mixed order for separator " + ", ".join(mixed)},
    ]
    return checks, spikes


class Store:
    def __init__(self, name, folder):
        self.name = name
        folder = Path(folder)
        orders, orders_report = load_orders(folder / "orders.parquet")
        users, users_report = load_users(folder / "users.csv")
        books, books_report = load_books(folder / "books.yaml")
        self.quality = {"orders": orders_report, "users": users_report, "books": books_report}

        person_of = users.set_index("user_id")["person_id"]
        orders["person_id"] = orders["user_id"].map(person_of).fillna(orders["user_id"]).astype(int)
        orphans = set(orders["user_id"]) - set(users["user_id"])
        orders_report["orders_with_unknown_user_id"] = int(orders["user_id"].isin(orphans).sum())
        orders_report["orders_with_unknown_book_id"] = int((~orders["book_id"].isin(books["book_id"])).sum())

        book_info = books.set_index("book_id")
        orders["title"] = orders["book_id"].map(book_info["title"]) if "title" in book_info else None
        orders["authors"] = orders["book_id"].map(book_info["authors"])
        self.orders, self.users, self.books = orders, users, books
        self._build_tables()
        self.metrics = self._metrics()
        self.checks, spikes = data_checks(
            self.daily_cents, self.orders["date"], orders_report, pd.Timestamp.today().normalize())
        self.metrics["data_checks"] = self.checks
        self.metrics["flagged_days"] = spikes

    def _build_tables(self):
        o, u, b = self.orders, self.users, self.books
        extra_o = [c for c in o.columns if c not in {
            "order_id", "user_id", "book_id", "quantity", "unit_price", "timestamp", "unit_price_raw",
            "price_currency", "unit_price_cents", "paid_cents", "paid_price", "date", "year", "month",
            "day", "person_id", "title", "authors"}]
        orders = pd.DataFrame({
            "order_id": o["order_id"].astype(int),
            "date": o["date"].dt.strftime("%Y-%m-%d"),
            "timestamp": o["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "year": o["year"].astype(int), "month": o["month"].astype(int), "day": o["day"].astype(int),
            "weekday": o["timestamp"].dt.day_name(),
            "user_id": o["user_id"].astype(int), "person_id": o["person_id"],
            "book_id": o["book_id"].astype(int), "title": o["title"], "authors": o["authors"],
            "quantity": o["quantity"], "unit_price": o["unit_price"], "paid_price": o["paid_price"],
            "price_currency": o["price_currency"], "unit_price_raw": o["unit_price_raw"],
        })
        for col in extra_o:
            orders[col] = o[col].values

        by_person = o.groupby("person_id").agg(
            orders=("order_id", "size"), books_bought=("quantity", "sum"), spent_cents=("paid_cents", "sum"),
            first_order=("date", "min"), last_order=("date", "max"))
        people = u.groupby("person_id").agg(
            user_ids=("user_id", lambda s: ", ".join(str(i) for i in sorted(s))),
            user_count=("user_id", "size"), names=("name", _join_unique), emails=("email", _join_unique),
            phones=("phone", _join_unique), addresses=("address", _join_unique))
        customers = people.join(by_person, how="outer")
        missing = customers["user_ids"].isna()
        customers.loc[missing, "user_ids"] = customers.index[missing].astype(str)
        customers["user_count"] = customers["user_count"].fillna(1).astype(int)
        customers["orders"] = customers["orders"].fillna(0).astype(int)
        customers["books_bought"] = customers["books_bought"].fillna(0).astype(int)
        customers["total_spent"] = customers["spent_cents"].fillna(0) / 100
        customers["first_order"] = customers["first_order"].dt.strftime("%Y-%m-%d")
        customers["last_order"] = customers["last_order"].dt.strftime("%Y-%m-%d")
        customers = customers.drop(columns="spent_cents").reset_index().rename(columns={"index": "person_id"})
        self.customers_cents = o.groupby("person_id")["paid_cents"].sum()

        extra_u = [c for c in u.columns if c not in {"user_id", "person_id", "name", "email", "phone", "address", "aliases", "name_key", "address_key", "phone_key"}]
        users = u[["user_id", "person_id", "name", "email", "phone", "address", "aliases"] + extra_u].copy()
        users["user_id"] = users["user_id"].astype(int)

        sold = o.groupby("book_id").agg(copies_sold=("quantity", "sum"), revenue_cents=("paid_cents", "sum"))
        books = b.drop(columns=["author_set"]).set_index("book_id").join(sold)
        books["copies_sold"] = books["copies_sold"].fillna(0).astype(int)
        books["revenue"] = books["revenue_cents"].fillna(0) / 100
        books = books.drop(columns="revenue_cents").reset_index()
        front = [c for c in ("book_id", "title", "authors", "author_count", "genre", "publisher", "year") if c in books]
        books = books[front + [c for c in books.columns if c not in front]]
        books["book_id"] = books["book_id"].astype(int)

        with_sets = b[b["author_count"] > 0][["book_id", "authors", "author_count"]].set_index("book_id").join(sold)
        author_sets = with_sets.groupby("authors").agg(
            author_count=("author_count", "first"), books=("author_count", "size"),
            copies_sold=("copies_sold", "sum"), revenue_cents=("revenue_cents", "sum"))
        author_sets["copies_sold"] = author_sets["copies_sold"].fillna(0).astype(int)
        author_sets["revenue"] = author_sets["revenue_cents"].fillna(0) / 100
        author_sets = author_sets.drop(columns="revenue_cents").reset_index()

        exploded = b[["book_id", "author_set"]].explode("author_set").dropna().rename(columns={"author_set": "author"})
        exploded = exploded.set_index("book_id").join(sold).reset_index()
        authors = exploded.groupby("author").agg(
            books=("book_id", "nunique"), copies_sold=("copies_sold", "sum"), revenue_cents=("revenue_cents", "sum"))
        authors["copies_sold"] = authors["copies_sold"].fillna(0).astype(int)
        authors["revenue"] = authors["revenue_cents"].fillna(0) / 100
        authors = authors.drop(columns="revenue_cents").reset_index()

        daily = o.groupby("date").agg(revenue_cents=("paid_cents", "sum"), orders=("order_id", "size"), copies=("quantity", "sum"))
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
        daily_revenue = pd.DataFrame({
            "date": daily.index.strftime("%Y-%m-%d"), "revenue": daily["revenue_cents"].values / 100,
            "orders": daily["orders"].values.astype(int), "copies": daily["copies"].values.astype(int)})

        self.daily_cents = daily["revenue_cents"]
        self.tables = {
            "orders": orders, "users": users, "customers": customers, "books": books,
            "authors": authors, "author_sets": author_sets, "daily_revenue": daily_revenue,
        }

    def table(self, name):
        if name not in self.tables:
            raise KeyError(name)
        return self.tables[name]

    def _metrics(self):
        t = self.tables
        top = self.daily_cents[self.daily_cents > 0].sort_values(ascending=False, kind="stable").head(5)
        top_days = [{"date": d.strftime("%Y-%m-%d"), "revenue": c / 100} for d, c in top.items()]

        sets = t["author_sets"]
        best_sets = sets[sets["copies_sold"] == sets["copies_sold"].max()] if len(sets) else sets
        authors = t["authors"]
        best_authors = authors[authors["copies_sold"] == authors["copies_sold"].max()] if len(authors) else authors

        spent = self.customers_cents
        best = spent[spent == spent.max()]
        users = self.users
        best_buyers = []
        for person_id, cents in best.items():
            ids = sorted(int(i) for i in users.loc[users["person_id"] == person_id, "user_id"]) or [int(person_id)]
            rows = users[users["user_id"].isin(ids)]
            best_buyers.append({
                "user_ids": ids, "total_spent": cents / 100,
                "names": sorted(set(rows["name"].dropna())), "emails": sorted(set(rows["email"].dropna())),
            })

        return {
            "store": self.name,
            "orders": int(len(self.orders)),
            "copies_sold": int(self.orders["quantity"].sum()),
            "total_revenue": int(self.orders["paid_cents"].sum()) / 100,
            "date_from": self.orders["date"].min().strftime("%Y-%m-%d"),
            "date_to": self.orders["date"].max().strftime("%Y-%m-%d"),
            "top_days": top_days,
            "unique_users": int(users["person_id"].nunique()),
            "user_records": int(len(users)),
            "unique_author_sets": int(len(sets)),
            "most_popular_author_sets": [
                {"authors": r.authors, "copies_sold": int(r.copies_sold)} for r in best_sets.itertuples()],
            "most_popular_individual_authors": [
                {"author": r.author, "copies_sold": int(r.copies_sold)} for r in best_authors.itertuples()],
            "best_buyers": best_buyers,
        }


class StoreRegistry:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._stores = {}
        self._lock = threading.Lock()

    def names(self):
        if not self.data_dir.is_dir():
            return []
        found = [p.name for p in self.data_dir.iterdir()
                 if p.is_dir() and all((p / f).exists() for f in ("orders.parquet", "users.csv", "books.yaml"))]
        return sorted(found, key=lambda n: (len(n), n))

    def get(self, name):
        if name not in self.names():
            raise KeyError(f"Unknown store '{name}'. Available: {', '.join(self.names())}")
        with self._lock:
            if name not in self._stores:
                self._stores[name] = self._load(name)
            return self._stores[name]

    def _load(self, name):
        folder = self.data_dir / name
        sources = [folder / f for f in ("orders.parquet", "users.csv", "books.yaml")] + [Path(__file__)]
        cache = self.data_dir.parent / ".cache" / f"{name}.pkl"
        if cache.exists() and cache.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
            try:
                with open(cache, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass
        store = Store(name, folder)
        try:
            cache.parent.mkdir(exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
        return store


def save_chart(store, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    daily = store.tables["daily_revenue"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(pd.to_datetime(daily["date"]), daily["revenue"], linewidth=1.4)
    flagged = daily[daily["date"].isin(store.metrics["flagged_days"])]
    if len(flagged):
        ax.scatter(pd.to_datetime(flagged["date"]), flagged["revenue"], color="red", zorder=3, label="suspicious spike")
        ax.legend()
    ax.set_title(f"{store.name}: daily revenue (USD)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue, $")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_report(store, out_dir):
    m = store.metrics
    print(f"\n=== {store.name} ===")
    print(f"Orders: {m['orders']:,} | revenue ${m['total_revenue']:,.2f} | {m['date_from']} .. {m['date_to']}")
    print("Top 5 days by revenue:")
    for i, d in enumerate(m["top_days"], 1):
        print(f"  {i}. {d['date']}  ${d['revenue']:,.2f}")
    print(f"Unique users: {m['unique_users']} (from {m['user_records']} user records)")
    print(f"Unique author sets: {m['unique_author_sets']}")
    for s in m["most_popular_author_sets"]:
        print(f"Most popular author set: {s['authors']} ({s['copies_sold']} copies)")
    for b in m["best_buyers"]:
        print(f"Best buyer: {b['user_ids']} (${b['total_spent']:,.2f})")
    for c in store.checks:
        print(f"Check [{c['status'].upper()}] {c['name']}: {c['detail']}")
    out_dir.mkdir(exist_ok=True)
    chart = out_dir / f"{store.name}_daily_revenue.png"
    save_chart(store, chart)
    (out_dir / f"{store.name}_results.json").write_text(
        json.dumps({"metrics": m, "data_quality": store.quality}, indent=2, default=str), encoding="utf-8")
    print(f"Chart: {chart}")
    print("Data quality:", json.dumps(store.quality, default=str))


def main(argv):
    root = Path(__file__).resolve().parent
    registry = StoreRegistry(root / "data")
    names = argv or registry.names()
    if not names:
        print("No stores found. Put STORE1 (orders.parquet, users.csv, books.yaml) into the data folder.")
        return 1
    for name in names:
        print_report(registry.get(Path(name).name), root / "output")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    import pipeline
    sys.exit(pipeline.main(sys.argv[1:]))
