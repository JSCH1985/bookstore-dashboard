# Bookstore sales dashboard

Cleans the `STORE1` data (`orders.parquet`, `users.csv`, `books.yaml`) with pandas, shows the results on a web dashboard, and adds a chat that answers questions about the data and controls the dashboard.

| File | Purpose |
|---|---|
| `pipeline.py` | Part 1: loading, cleaning, user reconciliation, metrics, matplotlib chart |
| `tools.py` | The API the AI calls: exact queries over the cleaned tables and dashboard control actions |
| `agent.py` | Tool-calling loop for any OpenAI-compatible model (Gemini, OpenAI, Groq…) plus the hallucination check |
| `app.py` | Web server (plain WSGI, no framework) |
| `static/index.html` | Dashboard and chat UI |

## Run on Windows (cmd)

```
cd /d %USERPROFILE%\Downloads
tar -xf bookstore-dashboard.zip
cd bookstore-dashboard
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python pipeline.py
copy .env.example .env
notepad .env
python agent.py "How many unique users are there?"
python app.py
```

`data\STORE1` … `data\STORE5` hold the original files. `python pipeline.py` prints all Part 1 answers and writes `output\STOREn_daily_revenue.png` and `output\STOREn_results.json` (`python pipeline.py STORE1` for one store).
`python app.py` opens the dashboard at http://localhost:8000 (Ctrl+C stops it); each `data\STORE*` folder gets its own tab at the top, with its own link (for example `/#STORE3`). Cleaned stores are cached in `.cache\` and rebuilt automatically when a data file or `pipeline.py` changes.

## Deploy

`render.yaml` deploys the app to a free Render web service: in Render choose New → Blueprint, pick this repository and paste the API key when asked for `LLM_API_KEY`.

## AI setup

Put a key into `.env`. The default is Google Gemini through its OpenAI-compatible endpoint (free key: https://aistudio.google.com/apikey). Any OpenAI-compatible provider works by changing `LLM_BASE_URL` and `LLM_MODEL`; `python agent.py --models` lists the model names your key can use.

## How the results are computed

- **Prices.** All 52 `unit_price` notations (`$ 35`, `30.0USD`, `€22¢25`, `46$99¢`, `USD 43.`, `52. €`…) are parsed with `Decimal`. `paid_price = quantity × unit_price`, converted at €1 = $1.2 and rounded to cents per order; the `unit_price` column shows the converted unit price rounded to cents. Money is summed as integer cents, so there is no float drift.
- **Timestamps.** Every format is normalized and parsed (ISO, `a.m./PM`, month names, `;`/`,` separators, time before or after the date, Unix time). For numeric `xx/xx/xx`, `xx.xx.xxxx` and `xx-xx-xxxx` dates the day/month order is detected from the data for each separator (a first part above 12 proves day-first) and printed in the cleaning report. `date`, `year`, `month`, `day` are extracted.
- **Cleaning.** Whitespace is collapsed, `NULL`/`-`/`N/A`/blank become missing, exact duplicate rows and repeated ids are removed, invalid book years become missing, phones are reduced to digits, e-mails lower-cased. Rows whose price, quantity, ids or timestamp cannot be recovered are dropped and counted. Everything is listed in the data cleaning report on the dashboard.
- **Real users.** Two user records are the same person when at least 3 of name, address, phone, e-mail match after normalization (only one field may change). Chains (A→B→C) are merged with union-find. Spending of all aliases is summed for the best buyer, who is shown with every user id.
- **Author sets.** `author` is split on commas into an unordered set, so `John, Paul` and `Paul, John` are one set and `John` alone is another. The most popular set is the one with the most copies sold (sum of `quantity`); ties are all shown.

## Data checks

A wrong timestamp parser does not fail loudly, it puts orders on the wrong day. The classic case: a plain `dateutil` parse reads `20:31,2025-09-20` as a time plus a timezone offset and takes the date from today, which piles 242 STORE1 orders onto the day you run it and turns today into the "best" day by far. So every store is checked after cleaning, and the result is shown under the totals on the dashboard:

- every order parsed, none dropped;
- no orders dated on or after the processing day;
- no day earning more than 3x its surrounding month (only days above twice the typical day are considered); flagged days are drawn red on the chart;
- the day/month order of numeric dates is unambiguous.

All four pass for all five stores; the busiest real day is at most 2.4x its month. Every parsed date was also compared with an independent strict parser: 0 differences in 49,545 timestamps.

## Results

| | STORE1 | STORE2 | STORE3 | STORE4 | STORE5 |
|---|---|---|---|---|---|
| Unique users (user records) | 3,221 (3,394) | 3,166 (3,319) | 2,955 (3,107) | 2,850 (3,008) | 2,987 (3,158) |
| Unique author sets | 324 | 281 | 307 | 299 | 311 |
| Most popular author set (copies) | Darron Ernser & Miss Lorina Kertzmann (173) | Kenneth Hamill & Marcus McKenzie (176) | Loreta Metz (155) | Emil Rempel & Heriberto Kutch & Merle Rice (198) | Dulcie Bode (169) |
| Best buyer ids | [48403, 48904, 50056] | [51014] | [52881, 53869, 54846] | [53347, 54809] | [52895, 53969] |
| Best buyer spent | $1,317.59 | $1,131.07 | $1,398.65 | $1,265.65 | $1,365.91 |
| Orders / revenue | 10,493 / $739,353.52 | 9,181 / $668,103.35 | 9,280 / $681,625.41 | 10,877 / $778,289.22 | 9,714 / $704,421.69 |

Top 5 days by revenue:

| # | STORE1 | STORE2 | STORE3 | STORE4 | STORE5 |
|---|---|---|---|---|---|
| 1 | 2025-08-20 $3,719.35 | 2025-11-16 $3,789.61 | 2025-10-09 $3,551.19 | 2025-11-24 $4,372.46 | 2025-10-20 $3,491.85 |
| 2 | 2025-09-22 $3,587.94 | 2025-12-12 $3,127.15 | 2025-09-21 $3,243.00 | 2025-11-17 $3,818.29 | 2025-11-28 $3,268.26 |
| 3 | 2026-01-15 $3,307.38 | 2025-12-10 $3,024.52 | 2025-09-04 $3,041.39 | 2025-11-28 $3,444.83 | 2025-09-27 $3,190.50 |
| 4 | 2025-09-03 $3,263.25 | 2025-09-19 $2,938.86 | 2025-09-30 $3,017.60 | 2025-11-13 $3,401.07 | 2025-10-17 $3,165.18 |
| 5 | 2025-10-17 $3,252.32 | 2025-11-19 $2,915.20 | 2025-09-01 $2,937.42 | 2025-09-21 $3,303.44 | 2025-12-05 $3,121.92 |

Every price and timestamp in all five stores was parsed; no order rows were dropped.

## How the chat avoids hallucinations

The model never sees the data. It has to call tools, and all counting, summing and averaging happens in pandas:

- `get_dashboard_metrics`, `describe_data`
- `query_table`: filters, group by, count, count_distinct, sum, mean, median, min, max over `orders`, `users`, `customers`, `books`, `authors`, `author_sets`, `daily_revenue`
- `distinct_values`, `find_customer`, `find_name_candidates`, `calculate` (safe arithmetic)
- dashboard control: `set_filter` (keep or exclude rows), `clear_filters`, `sort_table`, `show_view`, `set_chart`, `set_store`

After the model answers, every number in the answer is compared with the numbers in the tool results. If something does not match, the model is told which numbers are unsupported and must fetch them or remove them; if it still cannot, the answer is marked “unverified” in the chat. Each answer can be expanded to show the exact tool calls and results.

Commands such as “filter out books that have people names in the title” are handled in two steps: the model reads the actual titles (`distinct_values`, helped by `find_name_candidates`, which marks words that are first or last names of people in the data), decides which titles qualify, and the server applies an exact `title is one of [...]` filter and reports the row counts. Filters show as removable chips above the table.
