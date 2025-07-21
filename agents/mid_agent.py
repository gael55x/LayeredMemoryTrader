import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory

class MidTermAgent(BaseAgent):
    """
    Agent focusing on mid-term data to make trading decisions.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        super().__init__(name, config, semantic_memory)

    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes mid-term memory to decide on a trading action.
        """
        mid_term_data = memory_snapshot.get('mid_term')

        if mid_term_data is None or mid_term_data.empty or len(mid_term_data) < 20:
            return 'HOLD', 0.5

        # Example logic: Moving average crossover
        short_window = mid_term_data['close'].rolling(window=5).mean()
        long_window = mid_term_data['close'].rolling(window=20).mean()

        if short_window.iloc[-1] > long_window.iloc[-1] and short_window.iloc[-2] < long_window.iloc[-2]:
            return 'BUY', 0.7
        elif short_window.iloc[-1] < long_window.iloc[-1] and short_window.iloc[-2] > long_window.iloc[-2]:
            return 'SELL', 0.7
        else:
            return 'HOLD', 0.5
