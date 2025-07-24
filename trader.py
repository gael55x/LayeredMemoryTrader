import yaml
import pandas as pd
from data_manager import DataManager
from memory.memory_manager import MemoryManager
from memory.semantic_memory import SemanticMemory
from agents.short_agent import ShortTermAgent
from agents.mid_agent import MidTermAgent
from agents.long_agent import LongTermAgent
from agents.debate import Debate

def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

class Trader:
    def __init__(self, config_path='config.yaml', backtest_mode='train'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_manager = DataManager(self.config, backtest_mode=backtest_mode)
        self.memory_manager = MemoryManager(self.config['memory_horizons'])
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
            
            # --- Add Technical Indicators ---
            ticker_data['rsi'] = calculate_rsi(ticker_data)
            ticker_data['macd'], ticker_data['macd_signal'] = calculate_macd(ticker_data)
            ticker_data['upper_band'], ticker_data['lower_band'] = calculate_bollinger_bands(ticker_data)
            
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

                # Add news to semantic memory if available
                if 'news_summary' in current_data_slice.columns and not pd.isna(current_data_slice['news_summary'].iloc[-1]):
                    news_text = current_data_slice['news_summary'].iloc[-1]
                    if news_text != 'No significant news':
                        self.semantic_memory.add_memory(news_text)

                final_decision, final_confidence, votes = self.debate.run(memory_snapshot)

                # What this does:
                # If the confidence is high enough, it will make a decision to buy or sell.
                # If the confidence is not high enough, it will hold.
                # If the confidence is high enough, it will make a decision to buy or sell. 
                if final_confidence > self.config['thresholds']['mean_confidence_to_act']:
                    if final_decision == 'BUY' and self.portfolios[ticker]['cash'] > current_price:
                        # Proportional bet sizing
                        investment_amount = self.portfolios[ticker]['cash'] * final_confidence
                        shares_to_buy = investment_amount / current_price
                        self.portfolios[ticker]['shares'] += shares_to_buy
                        self.portfolios[ticker]['cash'] -= investment_amount
                        outcome = 'profit' # Simplified
                    elif final_decision == 'SELL' and self.portfolios[ticker]['shares'] > 0:
                        # Proportional selling
                        shares_to_sell = self.portfolios[ticker]['shares'] * final_confidence
                        self.portfolios[ticker]['cash'] += shares_to_sell * current_price
                        self.portfolios[ticker]['shares'] -= shares_to_sell
                        outcome = 'profit' # Simplified
                    else: # HOLD
                        outcome = 'neutral'
                else:
                    final_decision = 'HOLD' # Override decision if confidence is too low
                    outcome = 'neutral'
                
                # The outcome of a trade is only known when it's closed.
                # We will simplify the 'outcome' and focus the reflection on the 'why'.
                trade_outcome = "trade_executed" if final_decision != 'HOLD' else 'hold'
                
                # Update portfolio value history
                portfolio_value = self.portfolios[ticker]['cash'] + self.portfolios[ticker]['shares'] * current_price
                self.portfolios[ticker]['value_history'].append((current_data_slice.index[-1], portfolio_value))

                # Summarize agent votes for a more insightful reflection
                agent_votes_summary = ", ".join([f"{v['agent'].replace(' Agent', '')}: {v['decision']}({v['confidence']:.1f})" for v in votes])
                reflection_text = f"[{ticker}] Decision: {final_decision}, Conf: {final_confidence:.2f}. Votes: [{agent_votes_summary}]. Value: ${portfolio_value:,.2f}"
                
                self.memory_manager.add_reflection(
                    timestamp=current_data_slice.index[-1],
                    decision=final_decision,
                    confidence=final_confidence,
                    outcome=trade_outcome,
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
