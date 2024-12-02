import json
import logging
# Add the import for Dask DataFrame
from typing import Union
import pandas as pd
import dask.dataframe as dd  # Add this line
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator, llm  # Import llm function
import warnings
from langchain.llms import Cohere
from langchain import LLMChain, PromptTemplate
from .textgen_langchain import TextGeneratorLLM

logger = logging.getLogger("lida")

print("summarizer.py is being ifffe")  # to see if it's updated

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
            
Given the following detailed data description:

{data_description}

Your response should be a valid JSON object, and nothing else. Do not include any explanations or additional text. Use the following format:

```json
{{
    "dataset_name": "...",
    "dataset_description": "...",
    "fields": [
        {{
            "name": "...",
            "field_description": "...",
            "semantic_type": "...",
            "data_type": "...",
            "dtype": "...",
            "samples": ["...", "...", ...]
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
        
    def to_dict(self):
        return self.dict()
    
    def get_column_properties(self, df: Union[pd.DataFrame, dd.DataFrame], n_samples: int = 3) -> list:
        properties_list = []
        for column in df.columns:
            series = df[column]
            dtype = series.dtype
            is_dask = isinstance(df, dd.DataFrame)
            properties = {}
            # Determine data type
            if pd.api.types.is_numeric_dtype(dtype):
                properties["dtype"] = "number"
                num_unique = series.nunique()
                if num_unique.compute() if is_dask else num_unique > 20:
                    properties["data_type"] = "continuous"
                else:
                    properties["data_type"] = "discrete"
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                properties["dtype"] = "category"
                properties["data_type"] = "categorical"
            elif pd.api.types.is_bool_dtype(dtype):
                properties["dtype"] = "boolean"
                properties["data_type"] = "categorical"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                properties["dtype"] = "date"
                properties["data_type"] = "temporal"
            else:
                properties["dtype"] = str(dtype)
                properties["data_type"] = "unknown"

            # Add samples for each column
            non_null_values = series.dropna()
            if is_dask:
                non_null_values = non_null_values.compute()
            unique_values = non_null_values.unique()
            n_samples_actual = min(n_samples, len(unique_values))
            # Convert unique_values to a pandas Series before sampling
            samples = pd.Series(unique_values).sample(n=n_samples_actual, random_state=42).tolist()
            properties["samples"] = samples

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
            # Adjust sampling for Dask DataFrame
            if "samples" not in properties:
                if is_dask:
                    non_null_values = series.dropna().unique().compute()
                    n_samples_actual = min(n_samples, len(non_null_values))
                    samples = non_null_values[:n_samples_actual].tolist()
                else:
                    non_null_values = series.dropna().unique()
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
            self, data: Union[pd.DataFrame, dd.DataFrame, str],  # Include dd.DataFrame
            text_gen: TextGenerator = None, file_name="", n_samples: int = 3,
            textgen_config=TextGenerationConfig(n=1),
            summary_method: str = "default", encoding: str = 'utf-8') -> dict:
        # Adjust type checking to include Dask DataFrame
        if text_gen is not None:
            self.text_gen = text_gen

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding=encoding)
        
        # Correctly identify if data is a Dask DataFrame
        is_dask = isinstance(data, dd.DataFrame)

        # Sample data if it's a Dask DataFrame
        if is_dask:
            # Adjust the fraction as needed
            sample_frac = 0.01  # Sample 1% of the data
            sampled_data = data.sample(frac=sample_frac, random_state=42).compute()
        else:
            sampled_data = data

        # Generate column properties with new details
        data_properties = self.get_column_properties(sampled_data, n_samples)

        if summary_method == "langchain":
            if self.llm is None or self.summarization_chain is None:
                self._initialize_langchain()
            # Update LLM parameters based on textgen_config
            if textgen_config:
                self.llm.temperature = textgen_config.temperature
                self.llm.max_tokens = textgen_config.max_tokens
                if textgen_config.stop:
                    self.llm.stop = textgen_config.stop

            # Prepare detailed data description
            fields_info = []
            for prop in data_properties:
                field_info = {
                    "name": prop["column"],
                    "data_type": prop["properties"].get("data_type", ""),
                    "dtype": str(prop["properties"].get("dtype", "")),
                    "num_unique_values": prop["properties"].get("num_unique_values", 0),
                    "samples": prop["properties"].get("samples", []),
                }
                fields_info.append(field_info)

            data_description = {
                "dataset_name": file_name,
                "fields": fields_info,
            }
            data_description_str = json.dumps(data_description, indent=4)

            # Add logging to inspect the data description
            logger.debug(f"Data description being sent to LLM:\n{data_description_str}")

            # Create a new prompt template with correct input variable syntax
            summarization_prompt = PromptTemplate(
                input_variables=["data_description"],
                template=self.system_prompt + """
Given the following detailed data description:

{data_description}

Your response should be a valid JSON object, and nothing else. Do not include any explanations or additional text. Use the following format:

```json
{{
    "dataset_name": "...",
    "dataset_description": "...",
    "fields": [
        {{
            "name": "...",
            "field_description": "...",
            "semantic_type": "...",
            "data_type": "...",
            "dtype": "...",
            "samples": ["...", "...", ...]
        }},
        ...
    ],
    "name": "...",
    "file_name": "...",
    "field_names": ["...", "...", ...]
}}
""")

            # Initialize the LLMChain using the wrapped LLM
            summarization_chain = summarization_prompt | self.llm

            # Generate the summary using LLMChain
            summary_text = summarization_chain.invoke({"data_description": data_description_str})

            cleaned_summary_text = clean_code_snippet(summary_text)
            try:
                summary_data = json.loads(cleaned_summary_text)
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
            data_properties = self.get_column_properties(sampled_data, n_samples)

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

            return data_summary.to_dict()

    def _prepare_data_description(self, data: Union[pd.DataFrame, dd.DataFrame, str]) -> str:
        """
        Prepare the data description from the given data.

        Args:
            data (Union[pd.DataFrame, str]): The data to describe.

        Returns:
            str: String representation of the data description.
        """
        if isinstance(data, dd.DataFrame):
            # Sample a small fraction of data for summarization
            sampled_data = data.sample(frac=0.001, random_state=42).compute()
            return sampled_data.to_string(index=False)
        elif isinstance(data, pd.DataFrame):
            return data.head(5).to_string(index=False)
        elif isinstance(data, str):
            return data
        else:
            raise ValueError("Data must be a pandas DataFrame or a string.")
