import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory

class ShortTermAgent(BaseAgent):
    """
    Agent focusing on short-term data to make trading decisions, using an LLM for analysis.
    """
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes short-term memory using an LLM to decide on a trading action.
        """
        short_term_data = memory_snapshot.get('short_term')
        if short_term_data is None or short_term_data.empty:
            return 'HOLD', 0.5
            
        ticker = short_term_data['ticker'].iloc[-1]

        # Prepare the Prompt 
        prompt = f"You are a short-term momentum trader specializing in {ticker}. Based on the recent price action and technical indicators, what is your recommendation? Provide your answer as 'VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]'.\n\n"
        prompt += f"Short-Term Price & Indicator Data for {ticker} (last 10 data points):\n"
        prompt += short_term_data[['close', 'rsi', 'macd', 'upper_band', 'lower_band']].tail(10).to_string() + "\n"

        # Get LLM Response
        return self.ask_llm(prompt)
