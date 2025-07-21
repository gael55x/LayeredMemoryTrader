from abc import ABC, abstractmethod
import pandas as pd

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    """
    def __init__(self, name: str, config: dict):
        """
        Initializes the agent.
        
        :param name: The name of the agent (e.g., "Short-Term Agent").
        :param config: A configuration dictionary.
        """
        self.name = name
        self.config = config

    @abstractmethod
    def vote(self, memory_snapshot: dict) -> tuple[str, float]:
        """
        Analyzes the given memory snapshot and returns a trading decision and confidence score.
        
        :param memory_snapshot: A dictionary containing the 'short_term', 'mid_term',   
                                and 'long_term' memory DataFrames.
        :return: A tuple containing the vote ('BUY', 'SELL', 'HOLD') and a confidence score (0.0 to 1.0).
        """
        pass

if __name__ == '__main__':
    # This is an abstract class and cannot be instantiated directly.
    # The following code is for demonstration purposes of how a subclass would work.
    
    class DummyAgent(BaseAgent):
        def vote(self, memory_snapshot: dict) -> tuple[str, float]:
            print(f"Agent '{self.name}' is voting...")
            
            # Example logic: if the latest price in short-term memory is higher than the first, buy.
            short_term_data = memory_snapshot.get('short_term')
            if short_term_data is not None and not short_term_data.empty:
                if short_term_data['close'].iloc[-1] > short_term_data['close'].iloc[0]:
                    return 'BUY', 0.75
            
            return 'HOLD', 0.5

    # Example instantiation of a subclass
    dummy_config = {'some_param': 'value'}
    agent = DummyAgent(name="Dummy Agent", config=dummy_config)
    
    # Create a dummy memory snapshot
    dummy_memory = {
        'short_term': pd.DataFrame({
            'close': [100, 102, 101, 103]
        }),
        'mid_term': pd.DataFrame(),
        'long_term': pd.DataFrame()
    }
    
    decision, confidence = agent.vote(dummy_memory)
    print(f"Decision: {decision}, Confidence: {confidence}") 