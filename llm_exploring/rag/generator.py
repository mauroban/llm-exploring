from llm_exploring.models.llm import LLMManager


class RAGGenerator:
    def __init__(self, model_name: str):
        self.llm = LLMManager(model_name)
        self.llm.load_model()

    def generate(self, context: str, query: str) -> str:
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        return self.llm.generate_text(prompt, max_length=200)


# Example usage
if __name__ == "__main__":
    generator = RAGGenerator("gpt2")
    context = "The capital of France is Paris. It is known for the Eiffel Tower."
    query = "What is the capital of France?"
    response = generator.generate(context, query)
    print(f"Generated response: {response}")
