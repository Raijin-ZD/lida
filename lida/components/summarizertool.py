import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator
import warnings
import dask.dataframe as dd
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator  # Use validator instead of field_validator

print("SummarizerTool loaded")

class SummarizerTool(BaseTool, BaseModel):
    """
    A tool to summarize datasets using the provided text generator.
    """
    name: str = Field("summarizer", description="Summarizes datasets")
    description: str = Field("A tool to summarize datasets using the provided text generator.")
    text_generator: TextGenerator = Field(...)

    @validator('text_generator')
    def validate_text_generator(cls, v):
        if not isinstance(v, TextGenerator):
            raise ValueError('text_generator must be an instance of TextGenerator')
        return v

    def __init__(self, text_generator: TextGenerator):
        # Initialize the BaseTool with name and description
        super().__init__(name="summarizer", description="Summarizes datasets")
        self.text_generator = text_generator

    def _run(self, data: Union[pd.DataFrame, str], **kwargs) -> str:
        """
        Executes the summarization process.
        """
        summary = self.text_generator.generate(str(data))
        return f"Summary of the dataset: {summary}"

    async def _arun(self, data: Union[pd.DataFrame, str], **kwargs) -> str:
        """
        Asynchronous execution of the summarization process.
        """
        return await self._run(data, **kwargs)