import yaml
import pandas as pd
from data_manager import DataManager
from memory.memory_manager import MemoryManager
from agents.short_agent import ShortTermAgent
from agents.mid_agent import MidTermAgent
from agents.long_agent import LongTermAgent
from agents.debate import Debate

class Trader:
    def __init__(self, config_path='config.yaml', backtest_file='historical_data.csv'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_manager = DataManager(self.config, backtest_file=backtest_file)
        self.memory_manager = MemoryManager(self.config['memory_horizons'])

        # Initialize agents
        self.short_term_agent = ShortTermAgent(name="Short-Term Agent", config={})
        self.mid_term_agent = MidTermAgent(name="Mid-Term Agent", config={})
        self.long_term_agent = LongTermAgent(name="Long-Term Agent", config={})
        
        self.agents = [self.short_term_agent, self.mid_term_agent, self.long_term_agent]
        self.debate = Debate(self.agents)

    def run_backtest(self):
        print("Starting backtest...")
        while self.data_manager.fetch_data():
            current_data = self.data_manager.price_data
            if not current_data.empty:
                self.memory_manager.update_memory(current_data)
                memory_snapshot = self.memory_manager.get_memory_snapshot()

                final_decision, final_confidence, votes = self.debate.run(memory_snapshot)

                print(f"Timestamp: {current_data.index[-1]}")
                print(f"Votes: {votes}")
                print(f"Final Decision: {final_decision}, Confidence: {final_confidence}\n")

        print("Backtest finished.")

if __name__ == '__main__':
    trader = Trader()
    trader.run_backtest()
