# test_cohere.py

import cohere
import json
import re

# Replace '<your_cohere_api_key>' with your actual Cohere API key
COHERE_API_KEY = '3lEw7dPhMiInEez7n1CDJg0OumZNrqogGSeKfGaE'

co = cohere.Client(COHERE_API_KEY)

# Define the system prompt and the base summary
system_prompt = """
You are a helpful assistant that enriches dataset summaries by adding descriptions to fields. Given a dataset summary in JSON format, you will add detailed descriptions to the dataset and each field, based on their properties.

Instructions:
- For each field, analyze the properties and generate a concise description.
- The descriptions should be informative and helpful for understanding the dataset.
- Return the enriched summary in valid JSON format, including the new descriptions.
- Do not include any text outside the JSON object.

Example Input:
{
    "name": "Sample Dataset",
    "fields": [
        {
            "column": "age",
            "properties": {
                "dtype": "number",
                "min": 0,
                "max": 100,
                "samples": [23, 45, 67]
            }
        }
    ]
}

Example Output:
{
    "name": "Sample Dataset",
    "dataset_description": "This dataset contains demographic information.",
    "fields": [
        {
            "column": "age",
            "properties": {
                "dtype": "number",
                "min": 0,
                "max": 100,
                "samples": [23, 45, 67],
                "description": "Age of individuals ranging from 0 to 100."
            }
        }
    ]
}
"""

# Define your base summary (replace this with your actual data summary)
base_summary = {
    "name": "USA Cars Dataset",
    "fields": [
        {
            "column": "price",
            "properties": {
                "dtype": "number",
                "min": 1500,
                "max": 100000,
                "samples": [15000, 20000, 25000]
            }
        },
        {
            "column": "brand",
            "properties": {
                "dtype": "string",
                "samples": ["Ford", "Chevrolet", "Toyota"],
                "num_unique_values": 20
            }
        },
        # Add more fields as needed
    ]
}

# Create the prompt for the assistant
prompt = f"""
{system_prompt}

Annotate the dictionary below. Only return a JSON object.
{json.dumps(base_summary, indent=4)}
"""

# Call Cohere's generate endpoint
response = co.generate(
    model='command-xlarge-nightly',  # Use the model you have access to
    prompt=prompt,
    max_tokens=500,
    temperature=0.5,
    stop_sequences=["\n\n", "```"],
    return_likelihoods='NONE'
)

# Get the generated text
generated_text = response.generations[0].text.strip()
print("Generated Text:")
print(generated_text)

# Function to clean and extract JSON from the generated text
def extract_json(text):
    try:
        # Use regex to find JSON object in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in the response.")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error: {e}")

# Try to parse the generated text as JSON
try:
    enriched_summary = extract_json(generated_text)
    print("\nEnriched Summary:")
    print(json.dumps(enriched_summary, indent=4))
except ValueError as e:
    print("\nError parsing JSON:")
    print(e)
    print("\nRaw Response:")
    print(generated_text)
