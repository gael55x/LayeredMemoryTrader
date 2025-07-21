import pandas as pd
from trader import Trader

def evaluate_performance(trader: Trader):
    """
    Evaluates the performance of the trading bot after a backtest run.
    """
    reflections = trader.memory_manager.reflection_memory
    
    if reflections.empty:
        print("No trades were made during the backtest.")
        return

    print("Backtest Evaluation Results: \n")
    
    # --- Quantitative Analysis ---
    total_trades = len(reflections)
    wins = reflections[reflections['outcome'] == 'profit'].shape[0]
    losses = reflections[reflections['outcome'] == 'loss'].shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # --- Qualitative Analysis of Reflections ---
    print("\n Agent Reflections:")
    for index, row in reflections.iterrows():
        print(f"[{row['timestamp']}] Decision: {row['decision']} ({row['confidence']:.2f}) -> Outcome: {row['outcome']}")
        print(f"  Reflection: {row['reflection']}")

if __name__ == '__main__':
    # Run a backtest first
    print("Running backtest for evaluation...")
    trader_for_evaluation = Trader(backtest_mode='test')
    trader_for_evaluation.run_backtest()
    
    # Evaluate the results
    evaluate_performance(trader_for_evaluation)
