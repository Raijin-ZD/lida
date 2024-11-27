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
        self.api_key = api_key or os.getenv('COHERE_API_KEY')

        if self.model_type.lower() == 'cohere':
            if not self.api_key:
                raise ValueError("Cohere API key must be provided either via parameter or 'COHERE_API_KEY' environment variable.")
            self.llm = Cohere(
                model=self.model_name,
                cohere_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Define the prompt template
        self.summarization_prompt = PromptTemplate(
            input_variables=["data_description"],
            template="""
            You are an experienced data analyst that can annotate datasets.
            Your instructions are as follows:
            i) ALWAYS generate the name of the dataset and the dataset_description
            ii) ALWAYS generate a field description.
            iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g., company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc.
            You must return an updated JSON dictionary without any preamble or explanation.

            Summarize the following dataset:
            {data_description}
            """
        )

        # Initialize the LLMChain
        self.summarization_chain = LLMChain(llm=self.llm, prompt=self.summarization_prompt)

    def summarize(self, data: Union[pd.DataFrame, str]) -> str:
        """
        Generate a summary for the provided data.

        Args:
            data (Union[pd.DataFrame, str]): The dataset to summarize.

        Returns:
            str: JSON-formatted summary.
        """
        data_description = str(data)
        # Generate the summary using LLMChain
        summary = self.summarization_chain.run(data_description=data_description)
        return summary
