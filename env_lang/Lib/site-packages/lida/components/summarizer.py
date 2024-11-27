import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator
import warnings
import dask.dataframe as dd
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType

# Import the SummarizerTool from the new file
from .summarizertool import SummarizerTool  

system_prompt = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You must return an updated JSON dictionary without any preamble or explanation.
"""

logger = logging.getLogger("lida")
print("Summarizer loaded")

class Summarizer:
    def __init__(self, text_gen: TextGenerator):
        self.text_gen = text_gen

        self.summarization_prompt = PromptTemplate(
            input_variables=["data_description"],
            template="""
            Summarize the following dataset:
            {data_description}
            """
        )

        # Initialize the SummarizerTool correctly
        tools = [SummarizerTool(text_generator=self.text_gen)]
        self.agent = initialize_agent(
            tools,
            self.text_gen,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def summarize(self, data: Union[pd.DataFrame, str], **kwargs) -> str:
        return self.agent.run(data_description=str(data))
