# Layered Memory Trader

This repository implements a hierarchical memory multi-agent system inspired by "TradingGPT" ([arXiv:2309.03736](https://arxiv.org/abs/2309.03736)), but built for real-world crypto execution using a Gemini API key. The system is designed to be practical, explainable, and uses layered memory to have agents vote on trades. It does not require complex machine learning models or paid data subscriptions.

## Workflow Overview

1.  **Data Ingestion**
    -   Fetch BTC/USD price data from the Gemini public API.
    -   Pull sentiment data from free sources (e.g., Crypto Fear & Greed Index).

2.  **Memory Layers**
    -   **Short-term (1–2 hours):** Captures recent micro-trends.
    -   **Mid-term (1–2 days):** Tracks momentum and sentiment drift.
    -   **Long-term (weeks):** Identifies the macro market direction.

3.  **Agent Prompts**
    -   Each agent receives its corresponding memory chunk and a prompt to vote `BUY`, `HOLD`, or `SELL` with a confidence score.

4.  **Debate Coordinator**
    -   A weighted vote is conducted across all agent outputs.
    -   Action is taken if a configurable threshold is met (e.g., mean confidence ≥ 70%).

5.  **Execution**
    -   Places Dollar-Cost Averaging (DCA) buy or sell orders using the Gemini API.

6.  **Logging & Analysis**
    -   Records the memory context, agent votes, and the outcome of each trade.
    -   Evaluates the system's calibration by comparing agent confidence to profit and loss.

## Code Structure

-   **/data/:** Handles ingestion of price and sentiment data.
-   **/memory/:** Manages LLM memory buffers and decay logic.
-   **/agents/:**
    -   `short_agent.py`
    -   `mid_agent.py`
    -   `long_agent.py`
    -   `debate.py`: Aggregates votes and makes the final decision.
-   `trader.py`: Places trades via the Gemini API.
-   `evaluate.py`: For backtesting and analyzing agent calibration/PnL.
-   `config.yaml`: Stores settings for polling intervals, thresholds, and memory sizes.
-   **Sample Jupyter Notebook:** Demonstrates a 7-day simulated backtest and live logging in sandbox mode.

## Unique Features

-   **Layered-Memory Multi-Agent System:** A proven concept from "TradingGPT," applied to live crypto trading.
-   **Explainable AI for Traders:** Provides clear insight into why each agent voted a certain way, capturing trader perspectives across different time scales.
-   **Accessible:** Requires only a Gemini API key and uses free, public data sources. No ML training or data subscriptions are needed.
-   **Debate & Confidence Weighting:** Reveals the cognitive process behind trade decisions, as opposed to single-shot LLM commands.
-   **Educational:** A practical example of LLM memory management, prompt engineering, agent orchestration, and trading execution.
