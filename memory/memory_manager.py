import pandas as pd

class MemoryManager:
    def __init__(self, horizons: dict):
        self.horizons = horizons
        self.short_term_memory = pd.DataFrame()
        self.mid_term_memory = pd.DataFrame()
        self.long_term_memory = pd.DataFrame()
        self.reflection_memory = pd.DataFrame(columns=['timestamp', 'decision', 'confidence', 'outcome', 'reflection'])

    def update_memory(self, new_data: pd.DataFrame):
        """
        Updates all memory layers with new data and ensures they do not exceed their configured size.
        
        :param new_data: A DataFrame containing the new data points to add.
        """
        if new_data.empty:
            return
            
        # Update short-term memory
        self.short_term_memory = pd.concat([self.short_term_memory, new_data])
        if len(self.short_term_memory) > self.horizons['short_term']:
            self.short_term_memory = self.short_term_memory.iloc[-self.horizons['short_term']:]
            
        # Update mid-term memory
        self.mid_term_memory = pd.concat([self.mid_term_memory, new_data])
        if len(self.mid_term_memory) > self.horizons['mid_term']:
            self.mid_term_memory = self.mid_term_memory.iloc[-self.horizons['mid_term']:]
            
        # Update long-term memory
        self.long_term_memory = pd.concat([self.long_term_memory, new_data]).tail(self.horizons['long_term'])

    def add_reflection(self, timestamp, decision, confidence, outcome, reflection_text):
        new_reflection = pd.DataFrame({
            'timestamp': [timestamp],
            'decision': [decision],
            'confidence': [confidence],
            'outcome': [outcome],
            'reflection': [reflection_text]
        })
        self.reflection_memory = pd.concat([self.reflection_memory, new_reflection], ignore_index=True)

    def get_memory_snapshot(self) -> dict:
        """
        Returns a dictionary containing the current state of all memory layers.
        """
        return {
            'short_term': self.short_term_memory,
            'mid_term': self.mid_term_memory,
            'long_term': self.long_term_memory,
            'reflections': self.reflection_memory
        }

if __name__ == '__main__':
    import yaml

    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Example usage:
    memory_manager = MemoryManager(config['memory_sizes'])
    
    # Simulate adding data
    for i in range(200):
        dummy_data = pd.DataFrame(
            {'close': [100 + i]}, 
            index=[pd.to_datetime('now') + pd.Timedelta(minutes=i)]
        )
        memory_manager.update_memory(dummy_data)

    snapshot = memory_manager.get_memory_snapshot()
    
    print("Short-term memory size:", len(snapshot['short_term']))
    print(snapshot['short_term'].tail())
    
    print("\nMid-term memory size:", len(snapshot['mid_term']))
    print(snapshot['mid_term'].tail())

    print("\nLong-term memory size:", len(snapshot['long_term']))
    print(snapshot['long_term'].tail()) 