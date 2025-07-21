import pandas as pd
from agents.base_agent import BaseAgent

class ShortTermAgent(BaseAgent):
    """
    Agent focusing on short-term data to make trading decisions.
    """
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes short-term memory to decide on a trading action.
        """
        short_term_data = memory_snapshot.get('short_term')

        if short_term_data is None or short_term_data.empty:
            return 'HOLD', 0.5

        # Simple momentum strategy
        if short_term_data['close'].iloc[-1] > short_term_data['close'].iloc[0]:
            return 'BUY', 0.8
        elif short_term_data['close'].iloc[-1] < short_term_data['close'].iloc[0]:
            return 'SELL', 0.8
        else:
            return 'HOLD', 0.5
