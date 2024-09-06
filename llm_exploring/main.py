from llm_exploring.models.llm import LLMManager
from llm_exploring.models.fine_tuning import LLMFineTuner

# Initialize the LLMManager
llm_manager = LLMManager("gpt2")
llm_manager.load_model()

# Create a fine-tuner
fine_tuner = LLMFineTuner(llm_manager)

# Prepare your fine-tuning data
train_texts = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "The Eiffel Tower is located in Paris.",
    # Add more training examples here
]

# Fine-tune the model
fine_tuner.fine_tune(train_texts, output_dir="fine_tuned_model", num_train_epochs=3)

# Load the fine-tuned model
llm_manager.load_model(fine_tuned_path="fine_tuned_model")

# Generate text using the fine-tuned model
prompt = "The capital of France is"
generated_text = llm_manager.generate_text(prompt)
print(f"Generated text: {generated_text}")
