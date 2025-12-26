from collections import defaultdict
import time
import math

class CollectionGraph:
    def __init__(self):
        # Node weights: Importance of each collection
        self.node_weights = defaultdict(lambda: 1.0)
        
        # Edge weights: Co-occurrence strength (not fully used in simple routing but good for future)
        self.edge_weights = defaultdict(lambda: defaultdict(float))
        
        # Decay factor for time-based relevance
        self.decay_rate = 0.99
        
    def get_weight(self, collection_name: str) -> float:
        return self.node_weights[collection_name]
    
    def update(self, collection_name: str, reward: float = 0.1):
        """
        Update the weight of a collection based on user interaction or successful retrieval.
        """
        self.node_weights[collection_name] += reward
        
    def decay(self):
        """
        Apply decay to all weights to prioritize recent trends.
        """
        for col in self.node_weights:
            self.node_weights[col] *= self.decay_rate

# Global instance
graph_db = CollectionGraph()
