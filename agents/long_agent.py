import pandas as pd
from agents.base_agent import BaseAgent
from memory.semantic_memory import SemanticMemory

class LongTermAgent(BaseAgent):
    """
    Agent focusing on long-term data and macroeconomic trends.
    """
    def __init__(self, name: str, config: dict, semantic_memory: SemanticMemory):
        super().__init__(name, config, semantic_memory)

    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes long-term memory and semantic context to decide on a trading action.
        """
        long_term_data = memory_snapshot.get('long_term')

        if long_term_data is None or long_term_data.empty:
            return 'HOLD', 0.5

        # Base vote on overall trend
        if long_term_data['close'].is_monotonic_increasing:
            base_vote = 'BUY'
            base_confidence = 0.7
        elif long_term_data['close'].is_monotonic_decreasing:
            base_vote = 'SELL'
            base_confidence = 0.7
        else:
            base_vote = 'HOLD'
            base_confidence = 0.5

        # Use semantic memory to adjust vote
        try:
            semantic_results = self.semantic_memory.search_memory("market sentiment", k=3)
            if semantic_results:
                # Simple sentiment analysis: "positive" keywords boost BUY, "negative" boost SELL
                for result in semantic_results:
                    if any(keyword in result['text'] for keyword in ['positive', 'rally', 'hike']):
                        if base_vote == 'BUY':
                            base_confidence = min(1.0, base_confidence + 0.1)
                    elif any(keyword in result['text'] for keyword in ['negative', 'fears', 'war']):
                        if base_vote == 'SELL':
                            base_confidence = min(1.0, base_confidence + 0.1)
        except IndexError:
            # Not enough memories to search
            pass
            
        return base_vote, base_confidence
