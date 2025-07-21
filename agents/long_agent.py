import pandas as pd
from agents.base_agent import BaseAgent

class LongTermAgent(BaseAgent):
    """
    Agent focusing on long-term data and macroeconomic trends.
    """
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes long-term memory to decide on a trading action.
        """
        long_term_data = memory_snapshot.get('long_term')

        if long_term_data is None or long_term_data.empty:
            return 'HOLD', 0.5

        # Based on overall trend
        if long_term_data['close'].is_monotonic_increasing:
            return 'BUY', 0.9
        elif long_term_data['close'].is_monotonic_decreasing:
            return 'SELL', 0.9
        else:
            return 'HOLD', 0.6
