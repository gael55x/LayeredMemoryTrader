"""Replay original price-prompt construction offline against an eligible-prefix control.

Usage: python tests/reproduce_original_prompts.py REPOSITORY OUTPUT_DIRECTORY
No provider calls, embeddings, trades, or return calculations.
"""
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import yaml

PIN = "82ffbc6fc3306644ba40e871fdadf0b74d379e32"
repo, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
hashes = {}


def nodes(path):
    raw = subprocess.check_output(["git", "show", f"{PIN}:{path}"], cwd=repo)
    hashes[path] = hashlib.sha256(raw).hexdigest()
    return ast.parse(raw).body


def load(node):
    namespace = {"pd": pd, "re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-source", "exec"), namespace)
    return namespace[node.name]


Memory = load(next(n for n in nodes("memory/memory_manager.py") if isinstance(n, ast.ClassDef)))
indicators = {n.name: load(n) for n in nodes("trader.py") if isinstance(n, ast.FunctionDef)}
votes = {}
for agent in ("short", "mid", "long"):
    cls = next(n for n in nodes(f"agents/{agent}_agent.py") if isinstance(n, ast.ClassDef))
    votes[agent] = load(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "vote"))


def capture(agent, snapshot):
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return SimpleNamespace(text="VOTE: HOLD, CONFIDENCE: 0.5")

    stub = SimpleNamespace(model=SimpleNamespace(generate_content=generate),
                           semantic_memory=SimpleNamespace(search_memory=lambda *a, **k: []))
    with contextlib.redirect_stdout(io.StringIO()):
        votes[agent](stub, snapshot)
    assert len(prompts) <= 1
    return prompts[0] if prompts else None


def raw(path):
    value = subprocess.check_output(["git", "show", f"{PIN}:{path}"], cwd=repo)
    hashes[path] = hashlib.sha256(value).hexdigest()
    return value


config = yaml.safe_load(raw("config.yaml"))
data = pd.read_csv(io.BytesIO(raw("historical_data_with_news.csv")), parse_dates=["time"])
period = config["backtest"]["training_period"]
data = data[data["time"].between(period["start"], period["end"])]
original = Memory(config["memory_horizons"])
records, examples = [], {}
for ticker in data["ticker"].unique():
    rows = data[data["ticker"] == ticker].set_index("time")
    assert rows.index.is_unique and rows.index.is_monotonic_increasing
    rows["rsi"] = indicators["calculate_rsi"](rows)
    rows["macd"], rows["macd_signal"] = indicators["calculate_macd"](rows)
    rows["upper_band"], rows["lower_band"] = indicators["calculate_bollinger_bands"](rows)
    for i in range(5, len(rows), 5):
        prefix = rows.iloc[:i]
        original.update_memory(prefix)
        control = Memory(config["memory_horizons"])
        control.update_memory(prefix)
        actual, expected = original.get_memory_snapshot(), control.get_memory_snapshot()
        for agent in votes:
            before, clean = capture(agent, actual), capture(agent, expected)
            assert (clean is not None) == (agent != "mid" or i >= 20), (ticker, i, agent)
            selected = actual[agent + "_term"].tail(20 if agent == "mid" else 10)
            record = {"ticker": ticker, "i": i, "cutoff": str(prefix.index[-1].date()),
                      "agent": agent, "original_called": before is not None,
                      "control_called": clean is not None, "prompt_matches_control": before == clean,
                      "foreign_rows": int(selected["ticker"].ne(ticker).sum()) if before else 0,
                      "future_rows": int((selected.index > prefix.index[-1]).sum()) if before else 0,
                      "duplicate_rows": int(selected.reset_index().duplicated(["time", "ticker"]).sum()) if before else 0}
            records.append(record)
            if before != clean:
                examples.setdefault(agent, {"case": record, "original_prompt": before, "control_prompt": clean})
            if agent == "mid" and i >= 25 or agent != "mid" and i >= 10:
                assert before == clean, record

assert any(r["future_rows"] > 0 for r in records)
assert any(r["original_called"] and not r["control_called"] for r in records)
with (output / "decisions.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
summary = {
    "source_commit": PIN, "source_hashes": hashes, "pandas_version": pd.__version__,
    "decision_dates": len(records) // 3, "agent_opportunities": len(records),
    "original_prompts": sum(r["original_called"] for r in records),
    "control_prompts": sum(r["control_called"] for r in records),
    "different_prompt_or_call": sum(not r["prompt_matches_control"] for r in records),
    "prompts_with_future_rows": sum(r["future_rows"] > 0 for r in records),
    "per_agent": {a: {"changed": sum(not r["prompt_matches_control"] for r in records if r["agent"] == a),
                      "extra_early_calls": sum(r["original_called"] and not r["control_called"] for r in records if r["agent"] == a)} for a in votes},
    "examples": examples,
    "scope": "Original indicator functions, memory class and vote methods with capture-only model and empty semantic results. Full price prompt strings compared; semantic retrieval, model responses, fills and returns unmeasured. Control is an eligible-prefix oracle, not a production repair."
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ("source_hashes", "examples")}, indent=2))
