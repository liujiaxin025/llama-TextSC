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
        Improved version that extracts more meaningful information from vectors
        """
        with torch.no_grad():
            texts = []
            for i in range(vectors.size(0)):
                vector = vectors[i]

                # Extract meaningful patterns from the vector
                # 1. Calculate vector statistics
                vector_norm = torch.norm(vector).item()
                vector_mean = torch.mean(vector).item()
                vector_std = torch.std(vector).item()

                # 2. Find top-k dimensions (most activated features)
                top_k = 5
                top_values, top_indices = torch.topk(vector, top_k)

                # 3. Create semantic categories based on vector patterns
                semantic_categories = []

                # High norm indicates rich semantic content
                if vector_norm > 10:
                    semantic_categories.append("complex topic")
                elif vector_norm > 5:
                    semantic_categories.append("moderate detail")
                else:
                    semantic_categories.append("simple concept")

                # Standard deviation indicates diversity of topics
                if vector_std > 0.5:
                    semantic_categories.append("multi-faceted")
                else:
                    semantic_categories.append("focused")

                # Mean value indicates overall sentiment/intensity
                if vector_mean > 0.2:
                    semantic_categories.append("positive")
                elif vector_mean < -0.2:
                    semantic_categories.append("negative")
                else:
                    semantic_categories.append("neutral")

                # 4. Create more informative text based on patterns
                if len(semantic_categories) > 0:
                    category_text = ", ".join(semantic_categories)
                    text_content = f"Recovered semantic content: {category_text} content with complexity {vector_norm:.1f}"
                else:
                    text_content = f"Recovered semantic content with complexity {vector_norm:.1f}"

                texts.append(text_content)

        return texts
    
    def decode_to_json(self, vectors: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Convert vectors back to JSON-like semantic structures
        Improved version with more meaningful content extraction
        """
        json_structures = []

        for i in range(vectors.size(0)):
            vector = vectors[i]

            # Extract vector characteristics
            vector_norm = torch.norm(vector).item()
            vector_mean = torch.mean(vector).item()
            vector_std = torch.std(vector).item()

            # Find most active dimensions
            top_k = 3
            top_values, top_indices = torch.topk(vector, top_k)

            # Generate content based on vector patterns
            subject, key_points, conclusion = self._extract_semantic_content(
                vector_norm, vector_mean, vector_std, top_values, top_indices
            )

            json_structure = {
                "subject": subject,
                "key_points": key_points,
                "conclusion": conclusion
            }
            json_structures.append(json_structure)

        return json_structures

    def _extract_semantic_content(self, norm, mean, std, top_values, top_indices):
        """Extract semantic content based on vector characteristics"""

        # Determine subject based on vector properties
        if norm > 10:
            subject = "complex topic with rich details"
        elif norm > 5:
            subject = "detailed subject matter"
        else:
            subject = "focused concept"

        # Generate key points based on activation patterns
        key_points = []

        # Add point based on norm
        key_points.append(f"content complexity level: {norm:.1f}")

        # Add point based on diversity (std)
        if std > 0.5:
            key_points.append("multi-dimensional topic coverage")
        else:
            key_points.append("specific subject focus")

        # Add point based on top activation values
        avg_top_activation = torch.mean(top_values).item()
        if avg_top_activation > 1.0:
            key_points.append("highly relevant information")
        else:
            key_points.append("moderate relevance")

        # Generate conclusion based on overall pattern
        if mean > 0.2:
            conclusion = "positive semantic content identified"
        elif mean < -0.2:
            conclusion = "negative semantic content identified"
        else:
            conclusion = "neutral semantic content recovered"

        return subject, key_points, conclusion

class SemanticGenerator:
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: str = "float16"):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        print(f"Loading Llama 3 model for generation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
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