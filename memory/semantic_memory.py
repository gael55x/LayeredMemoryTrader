import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticMemory:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.entries = []

    def add_memory(self, text: str):
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding, dtype='float32'))
        self.entries.append(text)

    def search_memory(self, query_text: str, k: int = 5) -> list:
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    'text': self.entries[idx],
                    'distance': distances[0][i]
                })
        return results

if __name__ == '__main__':
    semantic_memory = SemanticMemory()
    semantic_memory.add_memory("Global tensions rise as trade war fears escalate.")
    semantic_memory.add_memory("Tech stocks rally on positive earnings reports.")
    semantic_memory.add_memory("Central bank announces interest rate hike.")

    search_results = semantic_memory.search_memory("What is the latest news on interest rates?")
    print(search_results) 