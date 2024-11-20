import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator
import warnings
import dask.dataframe as dd

system_prompt = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You must return an updated JSON dictionary without any preamble or explanation.
"""

logger = logging.getLogger("lida")
print("Summarizer loadedfffftt")

class Summarizer():
    def __init__(self) -> None:
        self.summary = None

    def check_type(self, dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: Union[pd.DataFrame, dd.DataFrame], n_samples: int = 3) -> list:
        """Get properties of each column with efficient sampling."""

        if isinstance(df, dd.DataFrame):
            # Sample a fraction without computing the entire DataFrame
            sample_fraction = 0.01  # Adjust as needed (e.g., 1% of data)
            df_sample = df.sample(frac=sample_fraction, random_state=42).compute()
        elif isinstance(df, pd.DataFrame):
            # Use the entire DataFrame without sampling
            df_sample = df
        else:
            raise TypeError("df must be a pandas or dask DataFrame")

        properties_list = []
        for column in df_sample.columns:
            dtype = df_sample[column].dtype
            properties = {}

            # Compute the number of unique values
            nunique = df_sample[column].nunique()

            # Process numeric columns
            if pd.api.types.is_numeric_dtype(dtype):
                stats = df_sample[column].describe()
                properties.update({
                    "dtype": "number",
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "max": float(stats['max'])
                })
            # Process boolean columns
            elif pd.api.types.is_bool_dtype(dtype):
                properties["dtype"] = "boolean"
            # Process datetime columns
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                properties["dtype"] = "date"
                properties["min"] = df_sample[column].min()
                properties["max"] = df_sample[column].max()
            else:
                # Determine if the column is categorical or string based on unique values
                properties["dtype"] = "category" if nunique / len(df_sample) < 0.5 else "string"

            # Get sample values
            samples = df_sample[column].dropna().unique()[:n_samples].tolist()
            properties.update({
                "samples": samples,
                "num_unique_values": int(nunique),
                "semantic_type": "",
                "description": ""
            })

            properties_list.append({"column": column, "properties": properties})

        return properties_list

    def enrich(self, base_summary: dict, text_gen: TextGenerator, textgen_config: TextGenerationConfig, retries=3) -> dict:
        """Enrich the data summary with descriptions and retry if incomplete response is received"""
        for attempt in range(retries):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": f"""
                Annotate the dictionary below. Only return a JSON object.
                {base_summary}
                """},
            ]

            response = text_gen.generate(messages=messages, config=textgen_config)
            try:
                json_string = clean_code_snippet(response.text[0]["content"])
                if json_string.strip() == '{':
                    raise ValueError("Received incomplete response from the language model")
                enriched_summary = json.loads(json_string)
                return enriched_summary  # Exit if successful
            except (ValueError, json.decoder.JSONDecodeError):
                if attempt < retries - 1:
                    print(f"Retrying... Attempt {attempt + 1}")
                    continue
                else:
                    # If failed after all retries, set a default value for enriched_summary
                    print("Fallback triggered: Setting default summary")
                    return base_summary

    def summarize(
            self, data: Union[pd.DataFrame, dd.DataFrame, str],
            text_gen: TextGenerator, file_name="", n_samples: int = 3,
            textgen_config=TextGenerationConfig(n=1),
            summary_method: str = "default", encoding: str = 'utf-8') -> dict:
        """Summarize data from a pandas or Dask DataFrame or a file location"""

        # If data is a file path, read it into a pandas or Dask DataFrame
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding=encoding)

        # Proceed to get column properties
        data_properties = self.get_column_properties(data, n_samples)

        # Default single stage summary construction
        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }

        data_summary = base_summary

        # Enrich with LLM if necessary
        if summary_method == "llm":
            data_summary = self.enrich(
                base_summary,
                text_gen=text_gen,
                textgen_config=textgen_config)
        elif summary_method == "columns":
            # No enrichment, only column names
            data_summary = {
                "name": file_name,
                "file_name": file_name,
                "dataset_description": ""
            }

        data_summary["field_names"] = data.columns.tolist()
        data_summary["file_name"] = file_name

        return data_summary
