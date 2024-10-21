from transformers import pipeline, set_seed
from summarizer import Summarizer
from lida.datamodel import TextGenerationConfig

# Define a wrapper class to use Hugging Face for text generation
class HuggingFaceTextGenerator:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline('text-generation', model=model_name)
        set_seed(42)  # Set a seed for reproducibility

    def generate(self, messages, config):
        # Use a simplified prompt for testing
        simplified_prompt = """
            Return a JSON object with a single key-value pair, where the key is 'number' and the value is 963143.
        """

        print(f"Sending simplified prompt: {simplified_prompt}")  # Debugging prompt

        # Generate text using the Hugging Face pipeline
        response = self.generator(simplified_prompt, max_length=50, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        
        print(f"Received raw response from Hugging Face model: {generated_text}")  # Debugging response
        return MockResponse(generated_text)

class MockResponse:
    def __init__(self, text):
        self.text = [{"content": text}]

# Instantiate the summarizer
summarizer = Summarizer()

# Use HuggingFaceTextGenerator for text generation
text_gen = HuggingFaceTextGenerator(model_name="gpt2")  # You can use other models like 'gpt2' or 'EleutherAI/gpt-neo-125M'
textgen_config = TextGenerationConfig(n=1)

# Path to the large dataset
file_path = "F:\eeeeee\lida\lida\components\large_test_data.csv"

# Summarize the dataset with LLM enrichment using Hugging Face
summary = summarizer.summarize(
    data=file_path,
    text_gen=text_gen,
    textgen_config=textgen_config,
    summary_method="llm",  # Use "llm" to test with Hugging Face enrichment
    encoding="utf-8"
)

# Print summary result
print(summary)
