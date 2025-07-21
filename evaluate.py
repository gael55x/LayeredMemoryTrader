import pandas as pd
from trader import Trader
import matplotlib.pyplot as plt
import os

def evaluate_performance(trader: Trader):
    """
    Evaluates the performance of the trading bot and saves the results.
    """
    results_dir = 'documentation/results'
    os.makedirs(results_dir, exist_ok=True)

    reflections = trader.memory_manager.reflection_memory
    
    if reflections.empty:
        print("No trades were made during the backtest.")
        return

    # --- Save Trade Log ---
    trade_log_path = os.path.join(results_dir, 'trade_log.csv')
    reflections.to_csv(trade_log_path, index=False)
    print(f"\nTrade log saved to {trade_log_path}")

    print("\n--- Backtest Evaluation Results ---")
    
    # --- Quantitative Analysis ---
    total_trades = len(reflections[reflections['decision'] != 'HOLD'])
    wins = reflections[reflections['outcome'] == 'profit'].shape[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # --- Plot Win/Loss Rate ---
    plt.figure(figsize=(8, 6))
    reflections['outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'grey'])
    plt.title('Trade Outcome Distribution')
    plt.ylabel('')
    win_loss_plot_path = os.path.join(results_dir, 'win_loss_pie_chart.png')
    plt.savefig(win_loss_plot_path)
    plt.close()
    print(f"Win/loss chart saved to {win_loss_plot_path}")

    # --- Plot Portfolio Performance ---
    plt.figure(figsize=(14, 8))
    for ticker, portfolio in trader.portfolios.items():
        if portfolio['value_history']:
            df = pd.DataFrame(portfolio['value_history'], columns=['time', 'value']).set_index('time')
            plt.plot(df.index, df['value'], label=f'{ticker} Portfolio Value')
    
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    portfolio_plot_path = os.path.join(results_dir, 'portfolio_performance.png')
    plt.savefig(portfolio_plot_path)
    plt.close()
    print(f"Portfolio performance graph saved to {portfolio_plot_path}")

if __name__ == '__main__':
    # Run a backtest first
    print("Running backtest for evaluation...")
    trader_for_evaluation = Trader(backtest_mode='test')
    trader_for_evaluation.run_backtest()
    
    # Evaluate the results
    evaluate_performance(trader_for_evaluation)
