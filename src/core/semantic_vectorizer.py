import torch
from sentence_transformers import SentenceTransformer
import json
from typing import Union, Dict, Any

class SemanticVectorizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
        print(f"Loaded Sentence-BERT model: {model_name}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
    def vectorize(self, semantic_json: Union[Dict[str, Any], str]) -> torch.Tensor:
        if isinstance(semantic_json, dict):
            # Convert JSON to string
            semantic_str = json.dumps(semantic_json, ensure_ascii=False)
        else:
            semantic_str = str(semantic_json)
            
        # Get embedding
        embedding = self.model.encode(semantic_str, convert_to_tensor=True)
        embedding = embedding.to(self.device)
        
        return embedding.unsqueeze(0)  # Add batch dimension
    
    def batch_vectorize(self, semantic_jsons: list) -> torch.Tensor:
        semantic_strs = []
        for item in semantic_jsons:
            if isinstance(item, dict):
                semantic_strs.append(json.dumps(item, ensure_ascii=False))
            else:
                semantic_strs.append(str(item))
        
        embeddings = self.model.encode(semantic_strs, convert_to_tensor=True)
        embeddings = embeddings.to(self.device)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()