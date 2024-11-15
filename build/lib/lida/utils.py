import base64
import json
import logging
from typing import Any, List, Tuple, Union
import os
import io
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tiktoken
from diskcache import Cache
import hashlib
import io
import dask.dataframe as dd


logger = logging.getLogger("lida")


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.

    :param col_name: The name of the column to be cleaned.
    :return: A sanitized string valid as a column name.
    """
    return re.sub(r'[^0-9a-zA-Z_]', '_', col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.

    :param df: The DataFrame with possibly dirty column names.
    :return: A copy of the DataFrame with clean column names.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]
    return cleaned_df



import os
import pandas as pd
import dask.dataframe as dd
from typing import Union

def read_dataframe(file_location: str, encoding: str = 'utf-8') -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Read a dataframe from a given file location and clean its column names.
    For large datasets (over 400MB), this function uses Dask to handle distributed loading.
    """
    file_extension = file_location.split('.')[-1]
    file_size_mb = os.path.getsize(file_location) / (1024 * 1024)  # File size in MB
    print(f"File size is {file_size_mb:.2f} MB.")

    read_funcs = {
        'json': lambda: pd.read_json(file_location, orient='records', encoding=encoding),
        'csv': lambda: pd.read_csv(file_location, encoding=encoding),
        'xls': lambda: pd.read_excel(file_location),
        'xlsx': lambda: pd.read_excel(file_location),
        'parquet': pd.read_parquet,
        'feather': pd.read_feather,
        'tsv': lambda: pd.read_csv(file_location, sep="\t", encoding=encoding)
    }

    if file_extension not in read_funcs:
        raise ValueError('Unsupported file type')

    # Decide on loading method based on file size
    try:
        if file_size_mb > 400:
            # For files over 400MB, use Dask for distributed loading
            if file_extension in ['csv', 'tsv']:
                print("Loading data as a Dask DataFrame.")
                df = dd.read_csv(file_location, encoding=encoding)
            elif file_extension == 'parquet':
                print("Loading data as a Dask DataFrame.")
                df = dd.read_parquet(file_location)
            elif file_extension in ['xls', 'xlsx']:
                print("Loading data as a Dask DataFrame.")
                df = dd.read_excel(file_location)
            else:
                raise ValueError('Unsupported file type for Dask loading')
        else:
            # For smaller files, load with Pandas
            df = read_funcs[file_extension]()
            print("Loading data as a Pandas DataFrame.")

    except Exception as e:
        print(f"Failed to read file: {file_location}. Error: {e}")
        raise

    # Clean column names
    cleaned_df = clean_column_names(df)

    # Return the DataFrame (either Pandas or Dask)
    return cleaned_df



def file_to_df(file_location: str):
    """ Get summary of data from file location """
    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df


def plot_raster(rasters: Union[str, List[str]], figsize: Tuple[int, int] = (10, 10)):
    """
    Plot a series of base64-encoded raster images in a horizontal layout.

    Args:
        rasters: A single base64 string or a list of base64-encoded strings representing the images.
        figsize: A tuple indicating the size of the figure to display.
    """
    plt.figure(figsize=figsize)

    if isinstance(rasters, str):
        rasters = [rasters]

    images = []

    # Find the max height for resizing
    max_height = 0
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format='PNG')

        max_height = max(max_height, image.shape[0])

    # Resize images to max_height while preserving the aspect ratio and alpha channel if it exists
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format='PNG')

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(max_height * aspect_ratio)
        image_resized = np.array([np.interp(np.linspace(
            0, len(row), new_width), np.arange(0, len(row)), row) for row in image])

        if image_resized.shape[2] == 4:  # If RGBA, preserve alpha channel
            alpha_channel = image_resized[:, :, 3:]
            # Drop the alpha for visualization
            image_resized = image_resized[:, :, :3]
            image_resized = np.clip(image_resized, 0, 1)
            image_resized = np.concatenate(
                (image_resized, alpha_channel), axis=2)

        images.append(image_resized)

    # Concatenate images along the width
    concatenated_image = np.concatenate(images, axis=1)

    plt.imshow(concatenated_image)
    plt.axis('off')
    plt.show()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def cache_request(cache: Cache, params: Any, values: Any = None) -> Any:
    # Generate a unique key for the request

    key = hashlib.md5(json.dumps(
        params, sort_keys=True).encode("utf-8")).hexdigest()
    # Check if the request is cached
    if key in cache and values is None:
        print("retrieving from cache")
        return cache[key]

    # Cache the provided values and return them
    if values:
        print("saving to cache")
        cache[key] = values
    return values


def clean_code_snippet(code_string):
    # Extract code snippet using regex
    cleaned_snippet = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', code_string)

    if cleaned_snippet:
        cleaned_snippet = cleaned_snippet.group(1)
    else:
        cleaned_snippet = code_string

    # remove non-printable characters
    # cleaned_snippet = re.sub(r'[\x00-\x1F]+', ' ', cleaned_snippet)

    return cleaned_snippet
