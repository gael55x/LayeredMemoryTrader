import pandas as pd
from trader import Trader
import matplotlib.pyplot as plt
import os
import numpy as np

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

    # Save Trade Log 
    trade_log_path = os.path.join(results_dir, 'trade_log.csv')
    reflections.to_csv(trade_log_path, index=False)
    print(f"\nTrade log saved to {trade_log_path}")

    print("\n Backtest Evaluation Results:")
    
    # Quantitative Analysis 
    total_trades = len(reflections[reflections['decision'] != 'HOLD'])
    
    print(f"Total Trades Executed: {total_trades}")
    
    # Plot Decision Distribution
    plt.figure(figsize=(8, 6))
    decision_counts = reflections['decision'].value_counts()
    decision_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon', 'lightgrey'])
    plt.title('Trade Decision Distribution (BUY/SELL/HOLD)')
    plt.ylabel('')
    decision_plot_path = os.path.join(results_dir, 'decision_distribution_pie_chart.png')
    plt.savefig(decision_plot_path)
    plt.close()
    print(f"Decision distribution chart saved to {decision_plot_path}")

    # Plot Portfolio Performance 
    plt.figure(figsize=(14, 8))
    for ticker, portfolio in trader.portfolios.items():
        if portfolio['value_history']:
            # Agent's performance
            df = pd.DataFrame(portfolio['value_history'], columns=['time', 'value']).set_index('time')
            plt.plot(df.index, df['value'], label=f'{ticker} Agent Portfolio')
            
            # Buy and Hold benchmark
            ticker_data = trader.data_manager.get_data_for_ticker(ticker)
            buy_and_hold_value = (10000 / ticker_data['close'].iloc[0]) * ticker_data['close']
            plt.plot(ticker_data.index, buy_and_hold_value, label=f'{ticker} Buy & Hold', linestyle='--')
    
    plt.title('Portfolio Value Over Time vs. Buy and Hold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    portfolio_plot_path = os.path.join(results_dir, 'portfolio_performance.png')
    plt.savefig(portfolio_plot_path)
    plt.close()
    print(f"Portfolio performance graph saved to {portfolio_plot_path}")

    # Advanced Metrics & Summary Report 
    report_content = "# Trading Strategy Performance Report\n\nThis report analyzes the performance of the agent-based trading strategy against a 'Buy and Hold' benchmark.\n\n"
    for ticker, portfolio in trader.portfolios.items():
        if portfolio['value_history']:
            # Correct Sharpe Ratio Calculation 
            # Create a daily-indexed DataFrame
            ticker_data = trader.data_manager.get_data_for_ticker(ticker)
            portfolio_df = pd.DataFrame(portfolio['value_history'], columns=['time', 'value']).set_index('time')
            daily_portfolio_values = portfolio_df.reindex(ticker_data.index).ffill()

            # Calculate daily returns from the properly sampled series
            daily_returns = daily_portfolio_values['value'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Max Drawdown 
            cumulative_max = daily_portfolio_values['value'].cummax()
            drawdown = (daily_portfolio_values['value'] - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            
            # Buy & Hold Performance 
            buy_and_hold_final_value = (10000 / ticker_data['close'].iloc[0]) * ticker_data['close'].iloc[-1]


            report_content += f"## {ticker} Performance\n"
            report_content += f"- **Agent Final Portfolio Value:** ${daily_portfolio_values['value'].iloc[-1]:,.2f}\n"
            report_content += f"- **Buy & Hold Final Value:** ${buy_and_hold_final_value:,.2f}\n"
            report_content += f"- **Sharpe Ratio (annualized daily):** {sharpe_ratio:.2f}\n"
            report_content += f"- **Max Drawdown:** {max_drawdown:.2%}\n\n"
            
            print(f"\n {ticker} Advanced Metrics:")
            print(f"Final Portfolio Value: ${daily_portfolio_values['value'].iloc[-1]:,.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")

    summary_path = os.path.join(results_dir, 'summary_report.md')
    with open(summary_path, 'w') as f:
        f.write(report_content)
    print(f"\nSummary report saved to {summary_path}")


if __name__ == '__main__':
    # Run a backtest first
    print("Running backtest for evaluation...")
    trader_for_evaluation = Trader(backtest_mode='test')
    trader_for_evaluation.run_backtest()
    
    # Evaluate the results
    evaluate_performance(trader_for_evaluation)
