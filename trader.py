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
        self.portfolios = {}

    def run_backtest(self):
        print("Starting backtest...")
        tickers = self.data_manager.tickers

        for ticker in tickers:
            print(f"\n--- Running backtest for {ticker} ---")
            self.portfolios[ticker] = {'cash': 10000, 'shares': 0, 'value_history': []}
            ticker_data = self.data_manager.get_data_for_ticker(ticker)
            
            if ticker_data.empty:
                print(f"No data for {ticker}, skipping.")
                continue

            # Iterate through the data for the current ticker
            for i in range(1, len(ticker_data)):
                # Run debate only once a week (every 5 trading days) 
                if i % 5 != 0:
                    continue

                current_price = ticker_data['close'].iloc[i]
                current_data_slice = ticker_data.iloc[:i]
                self.memory_manager.update_memory(current_data_slice)
                memory_snapshot = self.memory_manager.get_memory_snapshot()

                # Simulate adding a memory event (if news were available)
                if 'news' in current_data_slice.columns and not pd.isna(current_data_slice['news'].iloc[-1]):
                    self.semantic_memory.add_memory(current_data_slice['news'].iloc[-1])

                final_decision, final_confidence, votes = self.debate.run(memory_snapshot)

                # Simulate Portfolio Transaction 
                if final_decision == 'BUY' and self.portfolios[ticker]['cash'] > current_price:
                    shares_to_buy = self.portfolios[ticker]['cash'] / current_price
                    self.portfolios[ticker]['shares'] += shares_to_buy
                    self.portfolios[ticker]['cash'] = 0
                    outcome = 'profit' # Simplified for reflection
                elif final_decision == 'SELL' and self.portfolios[ticker]['shares'] > 0:
                    self.portfolios[ticker]['cash'] += self.portfolios[ticker]['shares'] * current_price
                    self.portfolios[ticker]['shares'] = 0
                    outcome = 'profit' # Simplified for reflection
                else: # HOLD
                    outcome = 'neutral'
                
                # Update portfolio value history
                portfolio_value = self.portfolios[ticker]['cash'] + self.portfolios[ticker]['shares'] * current_price
                self.portfolios[ticker]['value_history'].append((current_data_slice.index[-1], portfolio_value))

                reflection_text = f"[{ticker}] Decision: {final_decision}, Confidence: {final_confidence:.2f}, Portfolio Value: ${portfolio_value:,.2f}"
                
                self.memory_manager.add_reflection(
                    timestamp=current_data_slice.index[-1],
                    decision=final_decision,
                    confidence=final_confidence,
                    outcome=outcome,
                    reflection=reflection_text
                )

                if i % 100 == 0: # Print progress every 100 (processed) days
                    print(f"  Processed up to day {i} for {ticker}. Last decision: {final_decision}")

        print("\nBacktest finished.")

if __name__ == '__main__':
    # To run with training data
    trader_train = Trader(backtest_mode='train')
    trader_train.run_backtest()

    # To run with testing data
    # trader_test = Trader(backtest_mode='test')
    # trader_test.run_backtest()
