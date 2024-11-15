from dataclasses import asdict
from typing import Dict
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse

from ..scaffold import ChartScaffold
from lida.datamodel import Goal
import dask.dataframe as dd


system_prompt = system_prompt =  """
You are an expert data visualization assistant tasked with generating a complete, executable Python script to create a visualization based on the provided dataset summary and visualization goal.

**Requirements:**

1. **Write simple, clear code** using only basic, well-documented functions from the specified visualization library.
2. **Avoid using advanced or less-known features** unless absolutely necessary and you are certain they exist.
3. Handle both Pandas and Dask DataFrames.
   - If `data` is a Dask DataFrame and the library requires a Pandas DataFrame, convert it appropriately (e.g., using `data.compute()`).
4. Use appropriate methods for data preprocessing.
5. Create the visualization using the specified visualization library.
6. Return the visualization object (e.g., `fig`, `chart`, `img`) directly from the `plot` function.
7. **Do not include any explanations, comments, or extraneous text**.
8. Ensure all functions and variables used are defined within the code.

**Output:**

- Your output should be the complete Python code ready for execution.
- **Do not include any code fences (` ``` ), explanations, or extra text**.
- The code should start with the import statements.
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
