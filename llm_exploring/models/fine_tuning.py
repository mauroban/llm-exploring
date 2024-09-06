from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from .llm import LLMManager

class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class LLMFineTuner:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    def prepare_dataset(self, texts: List[str]) -> SimpleDataset:
        encodings = self.llm_manager.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        return SimpleDataset(encodings)

    def fine_tune(self, train_texts: List[str], output_dir: str, num_train_epochs: int = 3):
        if not self.llm_manager.model or not self.llm_manager.tokenizer:
            raise ValueError("Model and tokenizer not loaded. Call load_model() first.")

        train_dataset = self.prepare_dataset(train_texts)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.llm_manager.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

        # Save the fine-tuned model
        self.llm_manager.model.save_pretrained(output_dir)
        self.llm_manager.tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    llm_manager = LLMManager("gpt2")
    llm_manager.load_model()

    fine_tuner = LLMFineTuner(llm_manager)

    # Example fine-tuning data
    train_texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris.",
    ]

    fine_tuner.fine_tune(train_texts, output_dir="fine_tuned_model")
