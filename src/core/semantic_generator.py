import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, Any, Optional, List

class SemanticDeVectorizer:
    def __init__(self, vector_dim: int, hidden_dims: List[int], 
                 max_text_length: int = 256, device: str = "cuda"):
        self.device = device
        self.vector_dim = vector_dim
        self.max_text_length = max_text_length
        
        # Create a simple MLP decoder from vector to text embedding space
        layers = []
        prev_dim = vector_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs text tokens (we'll use a simple approach)
        layers.append(nn.Linear(prev_dim, max_text_length * 64))  # 64 is embedding dim
        
        self.decoder = nn.Sequential(*layers).to(device)
        self.text_embedding = nn.Linear(64, max_text_length * 1000).to(device)  # vocab size approximation
        
    def decode_to_text(self, vectors: torch.Tensor) -> List[str]:
        """
        Convert vectors back to semantic text representation
        This is a simplified version - in practice, you might want to use 
        a more sophisticated approach or train this jointly with the system
        """
        with torch.no_grad():
            # Decode vectors
            decoded = self.decoder(vectors)
            decoded = decoded.view(vectors.size(0), self.max_text_length, 64)
            
            # Convert to text embeddings (simplified)
            text_logits = self.text_embedding(decoded)
            
            # Simple greedy decoding to text (this is a placeholder)
            # In practice, you'd want to use a proper vocabulary and tokenizer
            texts = []
            for i in range(vectors.size(0)):
                # Create a simple JSON-like structure based on the vector
                vector_norm = torch.norm(vectors[i]).item()
                text_content = f"Semantic content with norm {vector_norm:.2f}"
                texts.append(text_content)
                
        return texts
    
    def decode_to_json(self, vectors: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Convert vectors back to JSON-like semantic structures
        """
        texts = self.decode_to_text(vectors)
        
        # Convert to JSON format
        json_structures = []
        for text in texts:
            # Create a simple JSON structure
            json_structure = {
                "subject": text[:30],
                "key_points": [text[:50]],
                "conclusion": text[:30]
            }
            json_structures.append(json_structure)
            
        return json_structures

class SemanticGenerator:
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: str = "float16"):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        print(f"Loading Llama 3 model for generation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.generation_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional writer. Your task is to expand the following key points from a communication system into a full, natural-sounding paragraph. Use your reasoning to fill in any gaps and correct potential minor errors in the provided information.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{semantic_info}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def generate(self, semantic_json: Dict[str, Any], max_length: int = 512) -> str:
        # Convert JSON to readable format
        semantic_str = json.dumps(semantic_json, indent=2, ensure_ascii=False)
        
        prompt = self.generation_prompt.format(semantic_info=semantic_str)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length + 256,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def batch_generate(self, semantic_jsons: List[Dict[str, Any]], 
                      batch_size: int = 4) -> List[str]:
        results = []
        for i in range(0, len(semantic_jsons), batch_size):
            batch = semantic_jsons[i:i + batch_size]
            batch_results = []
            for semantic_json in batch:
                result = self.generate(semantic_json)
                batch_results.append(result)
            results.extend(batch_results)
        return results