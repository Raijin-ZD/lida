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


    def get_column_properties(self, df: Union[pd.DataFrame, dd.DataFrame], n_samples: int = 3) -> list[dict]:
        """Get properties of each column in a pandas or Dask DataFrame"""

        #if isinstance(df, pd.DataFrame) and len(df) > 100000:
         #   df = dd.from_pandas(df, npartitions=10)
        # Remove the conversion; ensure data is loaded as Dask DataFrame initially

        # If the dataframe is a Dask dataframe, sample it for efficiency
      #  if isinstance(df, dd.DataFrame):
       #     if len(df) > 100000:
        #        df = df.sample(frac=0.1, random_state=42)  # Sample 10% of the data for large datasets

        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}

            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                # For Dask, compute the statistics after delayed calculations
                if isinstance(df, dd.DataFrame):
                  #Sampling for big datasets as .compute is expensive for larger datasets
                  sample_df = df.sample(frac=0.1, random_state=42) 
                  properties["std"] = self.check_type(dtype, sample_df[column].std().compute())
                  properties["min"] = self.check_type(dtype, sample_df[column].min().compute())
                  properties["max"] = self.check_type(dtype, sample_df[column].max().compute())
                else:
                    properties["std"] = self.check_type(dtype, df[column].std())
                    properties["min"] = self.check_type(dtype, df[column].min())
                    properties["max"] = self.check_type(dtype, df[column].max())

            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                # For datetime or categorical inference
                try:
                    if isinstance(df, dd.DataFrame):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd.to_datetime(df[column].compute(), errors='raise')
                        properties["dtype"] = "date"
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "date"
                except ValueError:
                    if isinstance(df, dd.DataFrame):
                        nunique_ratio = (df[column].nunique().compute() / len(df)).compute()
                    else:
                        nunique_ratio = df[column].nunique() / len(df)
                    
                    if nunique_ratio < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            # Add min/max if dtype is date
            if properties["dtype"] == "date":
                try:
                    if isinstance(df, dd.DataFrame):
                        properties["min"] = df[column].min().compute()
                        properties["max"] = df[column].max().compute()
                    else:
                        properties["min"] = df[column].min()
                        properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    if isinstance(cast_date_col, dd.Series):
                        properties["min"] = cast_date_col.min().compute()
                        properties["max"] = cast_date_col.max().compute()
                    else:
                        properties["min"] = cast_date_col.min()
                        properties["max"] = cast_date_col.max()

            # Add additional properties to the output dictionary
            #if isinstance(df, dd.DataFrame):
           #     nunique = df[column].nunique().compute()
          #      non_null_values = df[column].dropna().unique().compute()
         #   else:
        #        nunique = df[column].nunique()
       #         non_null_values = df[column].dropna().unique() 
            if isinstance(df, dd.DataFrame):
                # Approximate unique counts
                nunique = df[column].nunique_approx().compute()
                # Sample non-null values without computing all unique values
                sample_size = min(n_samples * 10, 1000)
                non_null_values = df[column].dropna().head(sample_size).compute().unique()
            else:
                nunique = df[column].nunique()
                non_null_values = df[column].dropna().unique()

            if "samples" not in properties:
                n_samples = min(n_samples, len(non_null_values))
                if isinstance(non_null_values, dd.Series):
                    samples = pd.Series(non_null_values).sample(n_samples, random_state=42).compute().tolist()
                else:
                    n_samples_actual = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(n=n_samples_actual, random_state=42).tolist()
                properties["samples"] = samples
                properties["num_unique_values"] = nunique
                properties["semantic_type"] = ""
                properties["description"] = ""
                properties_list.append({"column": column, "properties": properties})
                        #samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
                #properties["samples"] = samples
            #properties["num_unique_values"] = nunique
            #properties["semantic_type"] = ""
            #properties["description"] = ""
            #properties_list.append({"column": column, "properties": properties})
            
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
                    raise ValueError("Received incomplete response from Cohere")
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
                self, data: Union[pd.DataFrame, str],
            text_gen: TextGenerator, file_name="", n_samples: int = 3,
            textgen_config=TextGenerationConfig(n=1),
            summary_method: str = "default", encoding: str = 'utf-8') -> dict:
        """Summarize data from a pandas or Dask DataFrame or a file location"""

        # If data is a file path, read it into a pandas or Dask DataFrame
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding=encoding)
            
        # If data is a Dask DataFrame, trigger computation to convert it into a pandas DataFrame
        if isinstance(data, dd.DataFrame):
            data = data.compute()

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
