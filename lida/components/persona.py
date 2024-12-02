import json
import logging
from lida.utils import clean_code_snippet
from lida.datamodel import Persona, TextGenerationConfig
from langchain import LLMChain, PromptTemplate
from .textgen_langchain import TextGeneratorLLM  # Use relative import
from llmx import llm  # Import llm function
from dataclasses import asdict
import pandas as pd
import dask.dataframe as dd  # Add this line

logger = logging.getLogger("lida")

system_prompt = """You are an experienced data analyst who can take a dataset summary and generate a list of n personas (e.g., CEO or accountant for finance-related data, economist for population or GDP-related data, doctors for health data, or just users) that might be critical stakeholders in exploring some data and describe rationale for why they are critical. The personas should be prioritized based on their relevance to the data. Think step by step.
Add personas who would be particularly interested in large-scale data exploration, such as data scientists specialized in big data analytics.

Your response should be perfect JSON in the following format:
[{{"persona": "persona1", "rationale": "..."}}, {{"persona": "persona2", "rationale": "..."}}]
"""

class PersonaExplorer:
    """Generate personas given a summary of data"""

    def __init__(self, model_type, model_name, api_key):
        """
        Initialize the PersonaExplorer with specified model configuration.

        Args:
            model_type (str): Type of the model (e.g., 'cohere').
            model_name (str): Name of the model to use.
            api_key (str): API key for the model provider.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key

        # Initialize the TextGenerator with the specified provider
        self.text_gen = self._initialize_text_generator()

        # Wrap the TextGenerator with TextGeneratorLLM for LangChain compatibility
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=system_prompt)
        
        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["n", "summary"],
            template=system_prompt + "\nThe number of PERSONAs to generate is {{n}}. Generate {{n}} personas in the right format given the data summary below:\n{{summary}}"
        )

        # Update the persona chain initialization
        self.llm_chain = self.prompt_template | self.llm

    def _initialize_text_generator(self):
        """
        Initialize the TextGenerator with the specified provider.

        Returns:
            TextGenerator: An instance of a concrete TextGenerator subclass.
        """
        kwargs = {
            'provider': self.model_type,
            'api_key': self.api_key,
        }

        if self.model_type.lower() == 'cohere':
            kwargs['model'] = self.model_name  # Use 'model' for Cohere
        else:
            kwargs['model_name'] = self.model_name  # Use 'model_name' for other providers

        return llm(**kwargs)

    def generate(self, summary: dict, textgen_config: TextGenerationConfig, n=5) -> list[Persona]:
        """
        Generate personas given a summary of data.

        Args:
            summary (dict): Summary of the dataset.
            textgen_config (TextGenerationConfig): Configuration for text generation.
            n (int): Number of personas to generate.

        Returns:
            list[Persona]: A list of generated personas.
        """
        # Prepare variables for the prompt
        prompt_vars = {
            "n": n,
            "summary": json.dumps(summary, indent=4)
        }

        # Update LLM parameters based on textgen_config
        if textgen_config:
            self.llm.temperature = textgen_config.temperature
            self.llm.max_tokens = textgen_config.max_tokens
            if textgen_config.stop:
                self.llm.stop = textgen_config.stop

        try:
            # Generate the personas using the updated chain
            response = self.llm_chain.invoke(prompt_vars)
            logger.debug(f"Raw response from LLM: {response}")

            # Clean and parse the JSON output
            json_string = clean_code_snippet(response)
            result = json.loads(json_string)

            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]

            # Convert to list of Persona objects
            personas = [Persona(**x) for x in result]
            return personas

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            logger.error(f"Response received: {response}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate personas. "
                "Consider using a larger model or a model with higher max token length."
            ) from e
        except Exception as e:
            logger.error(f"An error occurred during persona generation: {e}")
            raise e
