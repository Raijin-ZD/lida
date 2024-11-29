import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator, llm  # Import llm function
import warnings
from langchain.llms import Cohere
from langchain import LLMChain, PromptTemplate
from .textgen_langchain import TextGeneratorLLM

logger = logging.getLogger("lida")

print("summarizer.py is being imported with summary update")  # to see if it's updated

system_prompt = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description.
ii) ALWAYS generate a field description.
iii) ALWAYS generate a semantic_type (a single word) for each field given its values e.g., company, city, number, supplier, location, gender, longitude, latitude, URL, IP address, zip code, email, etc.
You must return an updated JSON dictionary without any preamble or explanation.
"""

class Summarizer:
    def __init__(self, model_type=None, model_name=None, api_key=None, text_gen=None) -> None:
        """
        Initialize the Summarizer.

        Args:
            model_type (str): Type of the model (e.g., 'cohere').
            model_name (str): Name of the model to use.
            api_key (str): API key for the model provider.
            text_gen (TextGenerator): An instance of TextGenerator.
        """
        self.system_prompt = system_prompt
        self.summary = None
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.text_gen = text_gen

        # Initialize LangChain components if needed
        if self.model_type and self.model_name and self.api_key:
            self._initialize_langchain()
        else:
            self.llm = None
            self.summarization_chain = None

    def _initialize_langchain(self):
        """
        Initialize LangChain components.
        """
        # Initialize the TextGenerator with the specified provider
        if self.text_gen is None:
            self.text_gen = self._initialize_text_generator()

        # Wrap the TextGenerator with TextGeneratorLLM for LangChain compatibility
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=self.system_prompt)

        # Define the prompt template
        self.summarization_prompt = PromptTemplate(
            input_variables=["data_description"],
            template=self.system_prompt + """
            
Given the following data description:

{{data_description}}

Your response should be a valid JSON object, and nothing else. Do not include any explanations or additional text. Use the following format:

```json
{{
    "dataset_name": "...",
    "dataset_description": "...",
    "fields": [
        {{
            "name": "...",
            "field_description": "...",
            "semantic_type": "..."
        }},
        ...
    ],
    "name": "...",
    "file_name": "...",
    "field_names": ["...", "...", ...]
}}
""")

        # Initialize the LLMChain using the wrapped LLM
        self.summarization_chain = self.summarization_prompt | self.llm

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

        text_gen_instance = llm(**kwargs)
        return text_gen_instance

    def check_type(self, dtype: str, value):
        """Cast value to the correct type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            if pd.api.types.is_numeric_dtype(dtype):
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            elif pd.api.types.is_bool_dtype(dtype):
                properties["dtype"] = "boolean"
            elif pd.api.types.is_object_dtype(dtype):
                # Check if the string column can be cast to a valid datetime
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "date"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(dtype):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            # Add min and max if dtype is date
            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            # Add additional properties to the output dictionary
            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples_actual = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(
                    n_samples_actual, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            properties_list.append(
                {"column": column, "properties": properties})

        return properties_list

    def enrich(self, base_summary: dict, text_gen: TextGenerator,
               textgen_config: TextGenerationConfig) -> dict:
        """Enrich the data summary with descriptions"""
        logger.info("Enriching the data summary with descriptions")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"""
Annotate the dictionary below. Only return a JSON object.
{base_summary}
"""}
        ]

        response = text_gen.generate(messages=messages, config=textgen_config)
        enriched_summary = base_summary
        try:
            # Ensure response.text is a list of dictionaries and extract the 'content' field
            #content = response.text[0].get("content", "")
           # cleaned_summary = clean_code_snippet(content)
            json_string = clean_code_snippet(response.text[0]["content"])
            enriched_summary = json.loads(json_string)
        except json.decoder.JSONDecodeError:
            error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a default summary or a larger model with higher max token length. | {response.text[0]['content']}"
            logger.info(error_msg)
            print(response.text[0]["content"])
            raise ValueError(error_msg)
        return enriched_summary

    def summarize(
            self, data: Union[pd.DataFrame, str],
            text_gen: TextGenerator = None, file_name="", n_samples: int = 3,
            textgen_config=TextGenerationConfig(n=1),
            summary_method: str = "default", encoding: str = 'utf-8') -> dict:
        """Summarize data from a pandas DataFrame or a file location"""

        if text_gen is not None:
            self.text_gen = text_gen

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding=encoding)

        if summary_method == "langchain":
            if self.llm is None or self.summarization_chain is None:
                self._initialize_langchain()
            # Update LLM parameters based on textgen_config
            if textgen_config:
                self.llm.temperature = textgen_config.temperature
                self.llm.max_tokens = textgen_config.max_tokens
                if textgen_config.stop:
                    self.llm.stop = textgen_config.stop

            data_description = self._prepare_data_description(data)
            # Generate the summary using LLMChain
            summary_text = self.summarization_chain.invoke({"data_description": data_description})
            print("Summary Text:", summary_text)  # Add this line
            print("Type of summary_text:", type(summary_text))  # Check type
            print("Summary Text Content Structure:", {
            "is_dict": isinstance(summary_text, dict),
            "is_str": isinstance(summary_text, str),
            "length": len(str(summary_text))
            })
            cleaned_summary_text = clean_code_snippet(summary_text)  # Clean the output
            print("Cleaned Summary Text:", cleaned_summary_text)
            print("Type of Cleaned summary_text:", type(cleaned_summary_text))  # Check type
            print("Cleaned Summary Text Content Structure:", {
            "is_dict": isinstance(cleaned_summary_text, dict),
            "is_str": isinstance(cleaned_summary_text, str),
            "length": len(str(cleaned_summary_text))
            })
            try:
                summary_data = json.loads(cleaned_summary_text)
                print("Type of data:", type(summary_data))  # Check type
                print("data Text Content Structure:", {
                "is_dict": isinstance(summary_data, dict),
                "is_str": isinstance(summary_data, str),
                "length": len(str(summary_data))
             })
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
                logger.error(f"Summary Text: {summary_text}")
                raise ValueError("The summarizer did not return valid JSON.")
            # Ensure required fields are present in summary_data
            summary_data.setdefault("name", file_name)
            summary_data.setdefault("file_name", file_name)
            summary_data.setdefault("field_names", data.columns.tolist())
            return summary_data
        else:
            data_properties = self.get_column_properties(data, n_samples)

            # Default single-stage summary construction
            base_summary = {
                "name": file_name,
                "file_name": file_name,
                "dataset_description": "",
                "fields": data_properties,
            }

            data_summary = base_summary

            if summary_method == "llm":
                # Two-stage summarization with LLM enrichment
                data_summary = self.enrich(
                    base_summary,
                    text_gen=self.text_gen,
                    textgen_config=textgen_config)
            elif summary_method == "columns":
                # No enrichment, only column names
                data_summary = {
                    "name": file_name,
                    "file_name": file_name,
                    "dataset_description": "",
                    "fields": []
                }

            data_summary["field_names"] = data.columns.tolist()
            data_summary["file_name"] = file_name

            return data_summary

    def _prepare_data_description(self, data: Union[pd.DataFrame, str]) -> str:
        """
        Prepare the data description from the given data.

        Args:
            data (Union[pd.DataFrame, str]): The data to describe.

        Returns:
            str: String representation of the data description.
        """
        if isinstance(data, pd.DataFrame):
            return data.head(5).to_string(index=False)
        elif isinstance(data, str):
            return data
        else:
            raise ValueError("Data must be a pandas DataFrame or a string.")
