import yaml
from data_manager import DataManager
from memory.memory_manager import MemoryManager

class BacktestEngine:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_manager = DataManager(self.config, backtest_file='historical_data.csv')
        self.memory_manager = MemoryManager(self.config)

    def run(self):
        """
        Runs the backtest simulation.
        """
        print("Starting backtest...")
        
        # Main simulation loop
        while self.data_manager.fetch_data():
            # Get the latest data point
            latest_data = self.data_manager.price_data.tail(1)
            
            # Update the memory manager with the new data
            self.memory_manager.update_memory(latest_data)
            
            # Get a snapshot of the memory
            memory_snapshot = self.memory_manager.get_memory_snapshot()
            
            # TODO: This is where we will call the agents and the debate coordinator
            
            # For now, just print the current timestamp and memory sizes
            current_time = latest_data.index[0]
            print(f"Timestamp: {current_time} | "
                  f"Short: {len(memory_snapshot['short_term'])} | "
                  f"Mid: {len(memory_snapshot['mid_term'])} | "
                  f"Long: {len(memory_snapshot['long_term'])}")
                  
        print("Backtest finished.")

if __name__ == '__main__':
    engine = BacktestEngine(config_path='config.yaml')
    engine.run()
