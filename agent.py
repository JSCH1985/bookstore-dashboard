import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import tools

ROOT = Path(__file__).resolve().parent
MAX_STEPS = 12
MAX_TOOL_CHARS = 60000
NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

SYSTEM_PROMPT = """You are the assistant inside a bookstore sales dashboard. Current store: {store}. The dashboard tabs are overview, orders, users, customers and books; the user is looking at '{view}'.

You never see the data itself, only tool results.
Rules:
1. Every number, date, id, name or title you state must come from a tool result in this conversation. Never estimate. Never count, add, average or do any arithmetic yourself: use query_table (filters, group_by, aggregations) for counting and totals, and calculate for arithmetic on numbers from tool results.
2. Headline results (unique users, unique author sets, most popular author, best buyer, top revenue days, totals): call get_dashboard_metrics.
3. 'users' are raw user records. One real person (a customer) can own several user records with a changed name, address, phone or e-mail. Questions about real/unique users, people, customers or buyers use the customers table or get_dashboard_metrics.
4. Requests that change what the dashboard shows (filter, select, show only, hide, remove, sort, open a tab, zoom or regroup the chart, switch store) must be executed with the DASHBOARD CONTROL tools. 'select' / 'show only' = set_filter mode keep. 'filter out' / 'hide' / 'remove' / 'exclude' = mode exclude. Use exact conditions, e.g. op startswith with values ["s", "p"].
5. When a condition needs judgement the filter ops cannot express (for example book titles that contain a person's name), first fetch the values with distinct_values (find_name_candidates helps), decide which values qualify, then call set_filter with op "in" and the exact values. Briefly say which values you included.
6. If a tool returns an error, fix the arguments and retry. If the data cannot answer the question, say so.
7. Reply concisely in plain text, in the user's language. After a dashboard command, say what changed using the row counts from the tool result.

Tables (column: type):
{schema}"""


class LLMError(Exception):
    pass


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config():
    load_env()
    model = os.environ.get("LLM_MODEL", "gemini-flash-latest").strip()
    temperature = os.environ.get("LLM_TEMPERATURE")
    return {
        "api_key": (os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip(),
        "base_url": os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip().rstrip("/"),
        "model": model,
        "temperature": float(temperature) if temperature else (None if model.startswith("gemini") else 0.0),
    }


def _request(cfg, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(cfg["base_url"] + path, data=data, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 * (attempt + 1) ** 2)
                continue
            raise LLMError(f"HTTP {e.code} from {cfg['base_url']}: {body[:600]}")
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach {cfg['base_url']}: {e.reason}")
    raise LLMError("The AI service kept failing, try again later")


def complete(cfg, messages):
    payload = {"model": cfg["model"], "messages": messages, "tools": tools.schemas(), "tool_choice": "auto"}
    if cfg["temperature"] is not None:
        payload["temperature"] = cfg["temperature"]
    data = _request(cfg, "/chat/completions", payload)
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected response: {json.dumps(data)[:600]}")


def schema_text(ctx):
    lines = []
    for name, info in tools.describe_data(ctx)["tables"].items():
        cols = ", ".join(f"{c}: {t}" for c, t in info["columns"].items())
        lines.append(f"- {name} ({info['note']}): {cols}")
    return "\n".join(lines)


def _text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def numbers_in(text):
    text = re.sub(r"(?m)^\s*\d{1,2}[.)]\s", " ", text)
    return [(m.group(), float(m.group().replace(",", ""))) for m in NUMBER_RE.finditer(text)]


def collect_evidence(obj, acc):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        for raw, value in numbers_in(obj):
            acc.add(value)
        for digits in re.findall(r"\d+", obj):
            acc.add(float(digits))
    elif isinstance(obj, dict):
        for value in obj.values():
            collect_evidence(value, acc)
    elif isinstance(obj, (list, tuple)):
        acc.add(float(len(obj)))
        for value in obj:
            collect_evidence(value, acc)


def evidence_keys(values):
    keys = {"0.00", "1.00"}
    for v in values:
        for x in (v, v * 100):
            keys.update({f"{x:.2f}", f"{round(x, 1):.2f}", f"{round(x):.2f}"})
    return keys


def unverified_numbers(answer, evidence):
    keys = evidence_keys(evidence)
    return sorted({raw for raw, value in numbers_in(answer) if f"{value:.2f}" not in keys})


def _assistant_message(msg):
    out = {"role": "assistant"}
    if msg.get("content") is not None or not msg.get("tool_calls"):
        out["content"] = msg.get("content") or ""
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def chat(registry, state, history, message):
    cfg = config()
    names = registry.names()
    state = tools.sanitize_state(state, names)
    ctx = tools.ToolContext(registry, state)
    steps = []
    reply = {"steps": steps, "state": state, "model": cfg["model"]}
    if not names:
        return {**reply, "answer": "No data found: put STORE1 into the data folder.", "error": True}
    if not cfg["api_key"]:
        return {**reply, "answer": "The AI is not configured: set LLM_API_KEY in the .env file and restart.", "error": True}

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        store=state["store"], view=state["view"], schema=schema_text(ctx))}]
    for turn in (history or [])[-12:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"][:4000]})
    messages.append({"role": "user", "content": message})

    evidence = set()
    collect_evidence(message, evidence)
    retried = False
    try:
        for _ in range(MAX_STEPS):
            msg = complete(cfg, messages)
            calls = msg.get("tool_calls") or []
            messages.append(_assistant_message(msg))
            if calls:
                for call in calls:
                    fn = call.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args, result = {}, {"error": "arguments are not valid JSON"}
                    else:
                        result = tools.execute(fn.get("name"), args if isinstance(args, dict) else {}, ctx)
                    text = json.dumps(result, ensure_ascii=False, default=str)
                    if len(text) > MAX_TOOL_CHARS:
                        text = text[:MAX_TOOL_CHARS] + ' ... [truncated: narrow the query]'
                    if "error" not in result:
                        collect_evidence(result, evidence)
                    steps.append({"tool": fn.get("name"), "args": args, "result": text[:3000], "error": "error" in result})
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": text})
                continue
            answer = _text(msg.get("content")).strip()
            missing = unverified_numbers(answer, evidence)
            if missing and not retried:
                retried = True
                messages.append({"role": "user", "content": (
                    "Automatic check: these numbers in your answer do not appear in any tool result: "
                    f"{', '.join(missing)}. Get them with tools (query_table / calculate) or remove them, "
                    "then give the corrected final answer.")})
                continue
            return {**reply, "answer": answer or "(no answer)", "verified": not missing, "unverified_numbers": missing}
        return {**reply, "answer": "I could not finish within the step limit; try a narrower question.", "verified": False}
    except LLMError as e:
        return {**reply, "answer": f"AI service error: {e}", "error": True}


def main(argv):
    sys.stdout.reconfigure(errors="replace")
    cfg = config()
    if not cfg["api_key"]:
        print("LLM_API_KEY is not set (see .env.example)")
        return 1
    if argv and argv[0] == "--models":
        for m in _request(cfg, "/models").get("data", []):
            print(m.get("id"))
        return 0
    import pipeline
    registry = pipeline.StoreRegistry(ROOT / "data")
    question = " ".join(argv) or "How many unique users are there?"
    result = chat(registry, tools.default_state(registry.names()[0]), [], question)
    for step in result["steps"]:
        print(f"> {step['tool']}({json.dumps(step['args'], ensure_ascii=False)})")
    print(f"\n{result['answer']}\n\nverified: {result.get('verified')}  model: {cfg['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
