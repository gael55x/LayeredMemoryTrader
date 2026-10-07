"""Exercise the patched original replay method offline, including real FAISS reset.

python tests/check_memory_replay.py
Agent ask_llm is stubbed (provider boundary: tests/check_gemini_votes.py); no model, broker or network calls.
The evaluator runs on stub HOLD decisions only; no trading-return validation.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pandas as pd
import yaml

repo = Path(__file__).resolve().parents[1]
env = {"pd": pd, "faiss": faiss, "np": np, "re": re, "os": os, "plt": Mock(),
       "SentenceTransformer": lambda name: SimpleNamespace(encode=lambda texts: np.ones((len(texts), 384)))}
for path in ("memory/memory_manager.py", "memory/semantic_memory.py", "trader.py", "evaluate.py"):
    body = [n for n in ast.parse((repo / path).read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), env)
votes = {}
for agent in ("short", "mid", "long"):
    tree = ast.parse((repo / f"agents/{agent}_agent.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    vote = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "vote")
    exec(compile(ast.Module(body=[vote], type_ignores=[]), agent, "exec"), env)
    votes[agent] = env["vote"]


def capture(agent, snapshot):
    prompts = []
    def ask_llm(prompt):
        prompts.append(prompt)
        return "HOLD", 0.5
    stub = SimpleNamespace(ask_llm=ask_llm,
                           semantic_memory=SimpleNamespace(search_memory=lambda *a, **k: []))
    assert votes[agent](stub, snapshot) == ("HOLD", 0.5)
    assert len(prompts) <= 1
    return prompts

config = yaml.safe_load((repo / "config.yaml").read_text())
data = pd.read_csv(repo / "historical_data_with_news.csv", parse_dates=["time"])
period = config["backtest"]["training_period"]
data = data[data["time"].between(period["start"], period["end"])]
groups = {ticker: frame.set_index("time") for ticker, frame in data.groupby("ticker", sort=False)}


def run(frames, repeat=1):
    trader = env["Trader"].__new__(env["Trader"])
    trader.config, trader.portfolios = config, {}
    trader.memory_manager = env["MemoryManager"](config["memory_horizons"])
    trader.semantic_memory = env["SemanticMemory"]()
    shared, encoder, index = trader.semantic_memory, trader.semantic_memory.model, trader.semantic_memory.index
    signatures, state, prompt_counts = [], {}, []
    reference_calls, csv_logs = [], []
    working_add = trader.memory_manager.add_reflection

    def add_reflection(**row):
        reference_calls.append(row)
        return working_add(**row)

    trader.memory_manager.add_reflection = add_reflection

    def get(ticker):
        assert shared is trader.semantic_memory and shared.model is encoder and shared.index is index
        assert shared.entries == [] and index.ntotal == 0
        assert all(frame.empty for frame in trader.memory_manager.get_memory_snapshot().values())
        state.update(ticker=ticker, i=0)
        frame = frames[ticker].copy()
        state["frame"] = frame
        return frame

    def debate(snapshot):
        state["i"] += 5
        prefix = state["frame"].iloc[:state["i"]]
        for layer, horizon in config["memory_horizons"].items():
            actual = snapshot[layer]
            assert actual.equals(prefix.tail(horizon)), (state["ticker"], state["i"], layer)
            assert actual.index.is_unique and actual.index.is_monotonic_increasing
            assert actual["ticker"].eq(state["ticker"]).all()
        assert len(snapshot["reflections"]) == state["i"] // 5 - 1
        assert set(snapshot) == {"short_term", "mid_term", "long_term", "reflections"}
        assert not any(frame is trader.reflection_log for frame in snapshot.values())
        expected = {layer: prefix.tail(horizon) for layer, horizon in config["memory_horizons"].items()}
        for agent in votes:
            prompts = capture(agent, snapshot)
            assert prompts == capture(agent, expected), (state["ticker"], state["i"], agent)
            assert bool(prompts) == (agent != "mid" or state["i"] >= 20)
            prompt_counts.append(len(prompts))
        signatures.append((state["ticker"], state["i"], prefix.index[-1].isoformat()))
        return "HOLD", 0.5, []

    trader.data_manager = SimpleNamespace(tickers=list(frames), get_data_for_ticker=get)
    trader.debate = SimpleNamespace(run=debate)
    sequences = []
    for _ in range(repeat):
        signatures.clear()
        reference_calls.clear()
        trader.data_manager.get_data_for_ticker = get
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trader.run_backtest()
            reference = env["MemoryManager"](config["memory_horizons"])
            for row in reference_calls:
                reference.add_reflection(**row)
        assert trader.reflection_log.equals(reference.reflection_memory)
        assert trader.reflection_log.dtypes.equals(reference.reflection_memory.dtypes)
        expected_csv = reference.reflection_memory.to_csv(index=False)
        assert trader.reflection_log.to_csv(index=False) == expected_csv
        assert len(trader.reflection_log) == len(signatures)
        sequences.append(list(signatures))
        if len(signatures) == 537:
            # Run the real CSV exporter in isolation; plotting and model decisions are stubbed.
            trader.data_manager.get_data_for_ticker = lambda ticker: frames[ticker].copy()
            cwd = Path.cwd()
            with tempfile.TemporaryDirectory() as temp:
                try:
                    os.chdir(temp)
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(pd.Series, "plot", Mock()):
                        env["evaluate_performance"](trader)
                    log = Path("documentation/results/trade_log.csv")
                    assert log.exists(), "Evaluator lost all decisions after an empty final ticker"
                    csv_logs.append(log.read_text())
                    assert csv_logs[-1] == expected_csv
                    decisions = pd.read_csv(log)
                    assert len(decisions) == 537, "Evaluator dropped or duplicated run history"
                    counts = decisions["reflection"].str.extract(r"^\[([^]]+)\]", expand=False).value_counts().to_dict()
                    assert counts == {"AAPL": 179, "GOOG": 179, "MSFT": 179}, counts
                    assert list(decisions.columns) == ["timestamp", "decision", "confidence", "outcome", "reflection"]
                finally:
                    os.chdir(cwd)
    assert all(s == sequences[0] for s in sequences)
    assert all(csv == csv_logs[0] for csv in csv_logs)
    if len(signatures) == 537:
        assert sum(prompt_counts) == 1602 * repeat
    return sequences[0]


decisions = run({**groups, "EMPTY": pd.DataFrame()}, repeat=2)
assert len(decisions) == 537
for label, frame in {
    "unsorted": groups["AAPL"].iloc[::-1],
    "duplicate": pd.concat([groups["AAPL"].iloc[:1], groups["AAPL"]]),
}.items():
    try:
        run({"AAPL": frame})
    except ValueError as error:
        assert str(error) == "Timestamps for AAPL must be sorted and unique"
    else:
        raise AssertionError(label + " was accepted")

memory = env["MemoryManager"]({"short_term": 2, "mid_term": 3, "long_term": 4})
for i in range(6):
    memory.update_memory(groups["AAPL"].iloc[i:i + 1])
assert {k: len(memory.get_memory_snapshot()[k]) for k in memory.horizons} == memory.horizons
memory.reset()
assert all(f.empty for f in memory.get_memory_snapshot().values())
print(json.dumps({"decision_dates_per_run": len(decisions), "repeated_runs": 2,
                  "all_price_layers_match_eligible_prefix": True,
                  "price_prompts_per_run": 1602, "price_prompt_or_call_differences": 0,
                  "semantic_reset_real_faiss": True, "encoder_and_shared_identity_preserved": True,
                  "empty_ticker_skipped": True, "unsorted_and_duplicate_rejected": True,
                  "single_row_append_and_horizons_preserved": True,
                  "evaluation_records_after_repeat": 537, "evaluation_tickers": 3,
                  "scope": "Original patched replay method, memory classes, indicators and real FAISS; stub debate/encoder, actual evaluator CSV export; no provider constructors or financial performance validation."}, indent=2))
