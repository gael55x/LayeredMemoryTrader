"""Offline check of the Gemini vote boundary through the real google-genai SDK.

python tests/check_gemini_votes.py
HTTP is served by httpx.MockTransport; no credentials, network or billed calls.
Opt-in live smoke, one billed request with your own key:
GEMINI_API_KEY=... python tests/check_gemini_votes.py --live
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

import httpx
import pandas as pd
import yaml
from google import genai

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
# Keep the embedding model out of this provider-only check; no downloads or encoder calls.
sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = Mock(side_effect=AssertionError("encoder instantiated"))
sys.modules["sentence_transformers"] = sentence_transformers
from agents.base_agent import AgentVoteError
from agents.debate import Debate
from agents.long_agent import LongTermAgent
from agents.mid_agent import MidTermAgent
from agents.short_agent import ShortTermAgent
from memory.memory_manager import MemoryManager
import trader as trader_module

config = yaml.safe_load((repo / "config.yaml").read_text())
llm = config["llm"]
frame = pd.DataFrame({"ticker": "AAPL", "close": [100.0 + i for i in range(25)], "rsi": 50.0, "macd": 0.1,
                      "upper_band": 130.0, "lower_band": 95.0},
                     index=pd.date_range("2024-01-01", periods=25, name="time"))
snapshot = {"short_term": frame, "mid_term": frame, "long_term": frame}
semantic = SimpleNamespace(reset=lambda: None, add_memory=Mock(), search_memory=lambda *a, **k: [])


def make_agents():
    return [cls(name=name, config=llm, semantic_memory=semantic) for cls, name in (
        (ShortTermAgent, "Short-Term Agent"), (MidTermAgent, "Mid-Term Agent"), (LongTermAgent, "Long-Term Agent"))]


if sys.argv[1:] == ["--live"]:
    print(llm["model"], make_agents()[0].vote(snapshot))
    sys.exit()

KEY, SECRET = "offline-test-key", "PROVIDER-DETAIL-MUST-NOT-LEAK"
for name in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
    os.environ.pop(name, None)
os.environ["GEMINI_API_KEY"] = KEY
real_client, requests, clients, reply = genai.Client, [], [], {}


def handler(request):
    requests.append(request)
    if "raise" in reply:
        raise reply["raise"]
    return httpx.Response(reply.get("status", 200), json=reply["body"])


def text_body(text, finish="STOP"):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}]}


def offline_client(**kwargs):
    # Real SDK client and response parsing; only the HTTP transport is substituted.
    assert kwargs["api_key"] == KEY and kwargs["http_options"].timeout == llm["timeout_ms"]
    assert kwargs["http_options"].retry_options.attempts == 1
    assert kwargs["vertexai"] is False and kwargs["enterprise"] is False
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs["http_options"] = kwargs["http_options"].model_copy(update={"httpx_client": transport})
    client = real_client(**kwargs)
    owned_close = client.close
    def close():
        try:
            owned_close()
        finally:
            transport.close()  # The SDK leaves caller-injected transports for their caller to close.
    client.close = Mock(side_effect=close)
    clients.append(client)
    return client


def expect_abort(call):
    before, requests_before = len(clients), len(requests)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            call()
    except AgentVoteError as error:
        message = str(error)
        assert SECRET not in message and KEY not in message and "You are a" not in message, message
        assert error.__cause__ is None
        assert error.__context__ is None or error.__suppress_context__
        assert len(requests) - requests_before <= 1
        assert all(client.close.call_count == 1 for client in clients[before:])
        return message
    raise AssertionError("an invalid reply or provider failure was counted as a vote")


def reply_with(**case):
    reply.clear()
    reply.update(case)


valid = {"VOTE: BUY, CONFIDENCE: 0.7": ("BUY", 0.7), "vote: sell, confidence: 1": ("SELL", 1.0),
         "  Vote: Hold,\nConfidence: 0.0\n": ("HOLD", 0.0)}
malformed = ["VOTE: BUY, CONFIDENCE: 1.2", "VOTE: SELL, CONFIDENCE: -0.1", "VOTE: BUY, CONFIDENCE: nan",
             "VOTE: BUY, CONFIDENCE: inf", "VOTE: BUY, CONFIDENCE: " + "9" * 400,
             "VOTE: BUY, CONFIDENCE: 0.7 VOTE: SELL, CONFIDENCE: 0.9", "VOTE: BUY/SELL, CONFIDENCE: 0.5",
             "VOTE: [BUY], CONFIDENCE: [0.5]", "Sure. VOTE: BUY, CONFIDENCE: 0.7",
             "VOTE: BUY, CONFIDENCE: 0.75 because momentum", "VOTE: BUY", "", SECRET]
failures = {
    "auth": {"status": 401, "body": {"error": {"code": 401, "message": SECRET, "status": "UNAUTHENTICATED"}}},
    "server": {"status": 500, "body": {"error": {"code": 500, "message": SECRET, "status": "INTERNAL"}}},
    "timeout": {"raise": httpx.ReadTimeout(SECRET)},
    "blocked": {"body": {"promptFeedback": {"blockReason": "SAFETY"}}},
    "safety_stop": {"body": text_body("VOTE: BUY, CONFIDENCE: 0.7", finish="SAFETY")},
    "truncated": {"body": text_body("VOTE: BUY, CONFIDENCE: 0.", finish="MAX_TOKENS")},
    "no_text": {"body": {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]}},
}

with patch.object(genai, "Client", offline_client):
    for agent in make_agents():
        for text, expected in valid.items():
            reply_with(body=text_body(text))
            before = len(requests)
            with contextlib.redirect_stdout(io.StringIO()):
                assert agent.vote(snapshot) == expected, (agent.name, text)
            assert len(requests) == before + 1
            request = requests[-1]
            assert request.url.path.endswith(f"/models/{llm['model']}:generateContent"), request.url.path
            assert request.headers["x-goog-api-key"] == KEY
            assert request.extensions["timeout"]["read"] == llm["timeout_ms"] / 1000
            prompt = json.loads(request.content)["contents"][0]["parts"][0]["text"]
            assert prompt.startswith("You are a") and "VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]" in prompt
            assert clients[-1].close.call_count == 1
        for text in malformed:
            reply_with(body=text_body(text))
            expect_abort(lambda: agent.vote(snapshot))
        for label, case in failures.items():
            reply_with(**case)
            message = expect_abort(lambda: agent.vote(snapshot))
            assert label != "auth" or "401" in message, message

# Constructor failures must be sanitized too, before any request is made.
with patch.object(genai, "Client", Mock(side_effect=RuntimeError(SECRET))):
    expect_abort(lambda: make_agents()[0].vote(snapshot))

# Warmup votes need neither a key nor a client; a missing key stops Trader before data and encoder load.
with patch.dict(os.environ), patch.object(genai, "Client", Mock(side_effect=AssertionError("Gemini client created"))):
    del os.environ["GEMINI_API_KEY"]
    warmup = {"short_term": frame.iloc[:0], "mid_term": frame.iloc[:19], "long_term": frame.iloc[:0]}
    assert [agent.vote(warmup) for agent in make_agents()] == [("HOLD", 0.5)] * 3
    assert "GEMINI_API_KEY" in expect_abort(lambda: make_agents()[0].vote(snapshot))
    with patch.object(trader_module, "DataManager") as data_manager, \
            patch.object(trader_module, "SemanticMemory") as encoder:
        assert "GEMINI_API_KEY" in expect_abort(lambda: trader_module.Trader(config_path=str(repo / "config.yaml")))
        data_manager.assert_not_called()
        encoder.assert_not_called()

# Actual Trader.run_backtest with the actual Debate: valid HOLDs are logged, an invalid vote aborts unlogged.
trader = trader_module.Trader.__new__(trader_module.Trader)
trader.config, trader.portfolios = config, {}
trader.memory_manager = MemoryManager(config["memory_horizons"])
trader.semantic_memory = semantic
trader.data_manager = SimpleNamespace(tickers=["AAPL"], get_data_for_ticker=lambda ticker: frame.copy())
with patch.object(genai, "Client", offline_client):
    trader.debate = Debate(make_agents())
    reply_with(body=text_body("VOTE: HOLD, CONFIDENCE: 0.9"))
    before = len(requests)
    with contextlib.redirect_stdout(io.StringIO()):
        trader.run_backtest()
    # Days 5-20: short and long call each time, mid only once 20 rows exist.
    assert len(requests) - before == 9
    assert trader.reflection_log["decision"].tolist() == ["HOLD"] * 4
    reply_with(body=text_body("VOTE: BUY, CONFIDENCE: 1.2"))
    expect_abort(trader.run_backtest)
    assert trader.reflection_log.empty and trader.portfolios["AAPL"]["value_history"] == []

for path in ("requirements.txt", "agents/short_agent.py", "agents/mid_agent.py", "agents/long_agent.py"):
    assert "generativeai" not in (repo / path).read_text(), path
assert "google-genai==2.28.0" in (repo / "requirements.txt").read_text().split()
print(json.dumps({"model": llm["model"], "agents": 3, "valid_replies_per_agent": len(valid),
                  "malformed_aborts_per_agent": len(malformed), "provider_aborts_per_agent": len(failures),
                  "warmup_without_key_or_client": True, "missing_key_before_data_and_encoder": True,
                  "backtest_requests": 9, "abort_leaves_run_unlogged": True,
                  "scope": "Real google-genai request/response handling over httpx.MockTransport; no network or billed calls."},
                 indent=2))
