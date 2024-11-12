from dataclasses import asdict
from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ..scaffold import ChartScaffold
from lida.datamodel import Goal
import dask.dataframe as dd


system_prompt = system_prompt =  """
You are an expert data visualization assistant. Your task is to generate a complete, executable Python script that creates a visualization based on the provided dataset summary and visualization goal.

Requirements:
1. Inspect the dataset summary to identify the best columns for visualization based on their data types (e.g., numeric, categorical, datetime).
2. Write code that can handle both Pandas and Dask DataFrames.
3. If the data is a Dask DataFrame and the visualization library requires a Pandas DataFrame, sample a fraction and compute it to obtain a Pandas DataFrame.
4. Use appropriate methods for data preprocessing.
5. Create the visualization using the specified visualization library.
6. Return the visualization object (e.g., `fig`, `chart`, `img`) directly from the `plot` function.
7. Do not include any explanations or extraneous text outside of the code.

Your output should be the complete Python code ready for execution. Do not include any code fences, explanations, or extra text. The code should start with the import statements.
"""




class VizGenerator(object):
    """Generate visualizations from prompt"""

    def __init__(
        self
    ) -> None:

        self.scaffold = ChartScaffold()

    def generate(self, summary: Dict, goal: Goal,
                 textgen_config: TextGenerationConfig, text_gen: TextGenerator, library='seaborn'):
        """Generate visualization code given a summary and a goal"""
        # Optionally, select the library based on data size or user preference
      #  if len(summary['fields']) > 100000:
        #      library = 'datashader'

        library_template, library_instructions = self.scaffold.get_template(goal, library)
        messages = [
             {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"The dataset summary is: {summary}\n\n"},
        library_instructions,
        {"role": "user",
         "content": f"""
            Using the following template, generate the complete code for the visualization using {library}:
            {library_template}

            Remember:
            - Use only variables defined in the code or present in the dataset summary.
            - Dynamically select columns based on the dataset's columns and their data types.
            - Include data validation steps to ensure the code works with different datasets.
            - If the data is a Dask DataFrame and the library requires a Pandas DataFrame, sample a fraction and compute it to obtain a Pandas DataFrame.
            - Do not include any explanations, comments, or extra text.
            - The code should be ready for execution.

            The visualization goal is:
            {goal.question}

            Generate the code below:
              """
              }]

        completions: TextGenerationResponse = text_gen.generate(
            messages=messages, config=textgen_config)
        response = [x['content'] for x in completions.text]

        return response
