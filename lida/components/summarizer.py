import json
import logging
from typing import Union
import pandas as pd
from lida.utils import read_dataframe
from lida.datamodel import TextGenerationConfig
import warnings
import dask.dataframe as dd
from langchain.llms import Cohere
from langchain import LLMChain, PromptTemplate
import os  # For accessing environment variables
from dotenv import load_dotenv
from llmx import llm  # Import llm function
from .textgen_langchain import TextGeneratorLLM

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger("lida")
print("Summarizer loaded")

class Summarizer:
    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge-nightly', api_key: str = None):
        """
        Initialize the Summarizer with specified model configuration.

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
        self.llm = TextGeneratorLLM(text_gen=self.text_gen)

        # Define the prompt template
        self.summarization_prompt = PromptTemplate(
            input_variables=["data_description"],
            template="""
You are an experienced data analyst that can annotate datasets.
Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description.
ii) ALWAYS generate a field description.
iii) ALWAYS generate a semantic_type (a single word) for each field given its values, e.g., company, city, number, supplier, location, gender, longitude, latitude, URL, IP address, zip code, email, etc.
You must return an updated JSON dictionary without any preamble or explanation.

Summarize the following dataset:
{data_description}
            """
        )

        # Initialize the LLMChain using the wrapped LLM
        self.summarization_chain = LLMChain(llm=self.llm, prompt=self.summarization_prompt)

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

    def summarize(self, data: Union[pd.DataFrame, str], textgen_config: TextGenerationConfig = None) -> str:
        """
        Generate a summary for the provided data.

        Args:
            data (Union[pd.DataFrame, str]): The dataset to summarize.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.

        Returns:
            str: JSON-formatted summary.
        """
        if textgen_config:
            self.llm.temperature = textgen_config.temperature
            self.llm.max_tokens = textgen_config.max_new_tokens
            if textgen_config.stop:
                self.llm.stop = textgen_config.stop

        data_description = self._prepare_data_description(data)

        # Generate the summary using LLMChain
        summary = self.summarization_chain.run(data_description=data_description)
        return summary

    def _prepare_data_description(self, data: Union[pd.DataFrame, str]) -> str:
        """
        Prepare the data description from the given data.

        Args:
            data (Union[pd.DataFrame, str]): The data to describe.

        Returns:
            str: String representation of the data description.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=False)
        elif isinstance(data, str):
            return data
        else:
            raise ValueError("Data must be a pandas DataFrame or a string.")
