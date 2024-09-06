from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMManager:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self, fine_tuned_path: Optional[str] = None):
        print(f"Loading model {self.model_name}...")
        if fine_tuned_path:
            self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            self.model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set padding token to be the same as the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        print(f"Model loaded and moved to {self.device}")

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    # Use a small model for testing
    llm_manager = LLMManager("gpt2")
    llm_manager.load_model()

    prompt = "The quick brown fox"
    generated_text = llm_manager.generate_text(prompt)
    print(f"Generated text: {generated_text}")
