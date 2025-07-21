import yaml
import pandas as pd
from data_manager import DataManager
from memory.memory_manager import MemoryManager
from memory.semantic_memory import SemanticMemory
from agents.short_agent import ShortTermAgent
from agents.mid_agent import MidTermAgent
from agents.long_agent import LongTermAgent
from agents.debate import Debate

class Trader:
    def __init__(self, config_path='config.yaml', backtest_mode='train'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_manager = DataManager(self.config, backtest_mode=backtest_mode)
        self.memory_manager = MemoryManager(self.config['memory_sizes'])
        self.semantic_memory = SemanticMemory()

        # Initialize agents with semantic memory
        self.short_term_agent = ShortTermAgent(name="Short-Term Agent", config={}, semantic_memory=self.semantic_memory)
        self.mid_term_agent = MidTermAgent(name="Mid-Term Agent", config={}, semantic_memory=self.semantic_memory)
        self.long_term_agent = LongTermAgent(name="Long-Term Agent", config={}, semantic_memory=self.semantic_memory)
        
        self.agents = [self.short_term_agent, self.mid_term_agent, self.long_term_agent]
        self.debate = Debate(self.agents)

    def run_backtest(self):
        print("Starting backtest...")
        while self.data_manager.fetch_data():
            current_data = self.data_manager.price_data
            if not current_data.empty:
                self.memory_manager.update_memory(current_data)
                memory_snapshot = self.memory_manager.get_memory_snapshot()

                # Simulate adding a memory event
                if 'news' in current_data.columns and not pd.isna(current_data['news'].iloc[-1]):
                    self.semantic_memory.add_memory(current_data['news'].iloc[-1])

                final_decision, final_confidence, votes = self.debate.run(memory_snapshot)

                # Simulate trade outcome and reflection
                # In a real scenario, this would be determined by the market
                simulated_outcome = "profit" if final_decision == "BUY" else "loss" 
                reflection_text = f"The decision to {final_decision} resulted in a {simulated_outcome}. Confidence was {final_confidence:.2f}."
                
                self.memory_manager.add_reflection(
                    timestamp=current_data.index[-1],
                    decision=final_decision,
                    confidence=final_confidence,
                    outcome=simulated_outcome,
                    reflection=reflection_text
                )

                print(f"Timestamp: {current_data.index[-1]}")
                print(f"Votes: {votes}")
                print(f"Final Decision: {final_decision}, Confidence: {final_confidence}\n")

        print("Backtest finished.")

if __name__ == '__main__':
    # To run with training data
    trader_train = Trader(backtest_mode='train')
    trader_train.run_backtest()

    # To run with testing data
    # trader_test = Trader(backtest_mode='test')
    # trader_test.run_backtest()
