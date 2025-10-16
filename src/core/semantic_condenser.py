import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, Any, Optional

class SemanticCondenser:
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: str = "float16"):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        print(f"Loading Llama 3 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.condensation_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in information theory. Your task is to compress the user's text into its most essential semantic core, represented as a JSON object. The object must contain three keys: "subject", "key_points" (a list of strings), and "conclusion". The total word count of all values should be minimal while preserving the core meaning.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def condense(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        prompt = self.condensation_prompt.format(text=text)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length + 256,  # Extra space for response
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                semantic_json = json.loads(json_match.group())
                return semantic_json
            except json.JSONDecodeError:
                pass
        
        # Fallback: create simple structure
        return {
            "subject": text[:50] + "..." if len(text) > 50 else text,
            "key_points": [text[:100] + "..." if len(text) > 100 else text],
            "conclusion": text[:50] + "..." if len(text) > 50 else text
        }
    
    def batch_condense(self, texts: list, batch_size: int = 4) -> list:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            for text in batch:
                result = self.condense(text)
                batch_results.append(result)
            results.extend(batch_results)
        return results