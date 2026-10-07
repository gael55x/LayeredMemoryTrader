import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory

class MidTermAgent(BaseAgent):
    """
    Agent focusing on mid-term data to make trading decisions, using an LLM for analysis.
    """
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes mid-term memory using an LLM to decide on a trading action.
        """
        mid_term_data = memory_snapshot.get('mid_term')
        if mid_term_data is None or mid_term_data.empty or len(mid_term_data) < 20:
            return 'HOLD', 0.5
            
        ticker = mid_term_data['ticker'].iloc[-1]

        # Prepare the Prompt 
        prompt = f"You are a mid-term trend analyst specializing in {ticker}. Based on the following price data and technical indicators, what is your recommendation? Provide your answer as 'VOTE: [BUY/SELL/HOLD], CONFIDENCE: [0.0-1.0]'.\n\n"
        
        # Add mid-term price trend with moving averages and RSI
        prompt += f"Mid-Term Price & Indicator Data for {ticker} (last 20 data points):\n"
        prompt += mid_term_data[['close', 'rsi', 'macd', 'upper_band', 'lower_band']].tail(20).to_string() + "\n\n"
        prompt += "5-day Moving Average:\n"
        prompt += mid_term_data['close'].rolling(window=5).mean().tail().to_string() + "\n\n"
        prompt += "20-day Moving Average:\n"
        prompt += mid_term_data['close'].rolling(window=20).mean().tail().to_string() + "\n"

        # Get LLM Response
        return self.ask_llm(prompt)
