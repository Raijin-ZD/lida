from dataclasses import asdict

from lida.datamodel import Goal
print("Generating visualization code scaffold...")
class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(self) -> None:
        pass

    def get_template(self, goal: Goal, library: str):

        general_instructions = f"""
**Instructions for Generating the Visualization Code:**

1. **Write simple and clear code** using only basic functions from the specified library ({library}).
2. **Avoid using advanced or less-known features** unless you are certain they exist and are necessary.
3. **Use functions and methods that are well-documented and widely used**.
4. **Do not use any functions or methods unless you are sure they exist in the library's API**.

Given the dataset summary and the visualization goal, the `plot(data)` function should generate a {library} chart ({goal.visualization}) that addresses the goal: {goal.question}.

**Data Handling:**
- The data is already loaded and available in the variable `data`.
- The code should handle both Pandas and Dask DataFrames.
- If `data` is a Dask DataFrame and the library requires a Pandas DataFrame, convert it appropriately (e.g., using `data.compute()`).

**General Guidelines:**
- **Avoid unnecessary complexity**.
- **Ensure all variables and functions used are defined within the code**.
- **Do not include any explanations, comments, or extra text outside of the code**.

**Output:**
- The `plot(data)` function should return the visualization object (e.g., `fig`, `chart`, `img`).
"""

        matplotlib_instructions = f"""
{general_instructions}

**Additional Matplotlib/Seaborn Instructions:**
1. Use only well-documented matplotlib/seaborn functions
2. Set appropriate figure size using plt.figure() when needed
3. Add proper labels and titles
4. Use color palettes appropriately
5. Handle axis formatting and scaling as needed
"""

        if library == "matplotlib":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions} Use Basemap for charts that require a map."
            }
            template = f"""
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
<imports>
def plot(data):
    # Check if data is a Dask DataFrame and sample if necessary
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    <stub>  # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt

chart = plot(data)  # Data is already loaded. No additional code beyond this line.
"""
        elif library == "seaborn":
            instructions = {
                "role": "assistant",
                "content": f"{matplotlib_instructions} Use appropriate seaborn specific functions when possible."
            }
            template = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
<imports>
def plot(data):
    # Check if data is a Dask DataFrame and sample if necessary
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    <stub>  # only modify this section
    plt.title('{goal.question}', wrap=True)
    return plt

chart = plot(data)  # Data is already loaded. No additional code beyond this line.
"""
        elif library == "ggplot":
            instructions = {
                "role": "assistant",
                "content": f"{general_instructions} The `plot` method must return a ggplot object (`chart`). Think step by step."
            }
            template = f"""
import plotnine as p9
import pandas as pd
import dask.dataframe as dd
<imports>
def plot(data):
    # Check if data is a Dask DataFrame and sample if necessary
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    chart = <stub>

    return chart

chart = plot(data)  # Data is already loaded. No additional code beyond this line.
"""
        elif library == "altair":
            instructions = {
                "role": "system",
                "content": f"""{general_instructions} Always add a type that is BASED on `semantic_type` to each field such as `:Q`, `:O`, `:N`, `:T`, `:G`. Use `:T` if `semantic_type` is year or date. The `plot` method must return an Altair chart object (`chart`). Think step by step."""
            }
            template = f"""
import altair as alt
import pandas as pd
import dask.dataframe as dd
<imports>
def plot(data):
    # Check if data is a Dask DataFrame and sample if necessary
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    <stub>  # only modify this section
    return chart

chart = plot(data)  # Data is already loaded. No additional code beyond this line.
"""
        elif library == "plotly":
            instructions = {
                "role": "system",
                "content": f"""{general_instructions} If calculating metrics such as mean, median, mode, etc., ALWAYS use the option `numeric_only=True` when applicable and available. AVOID visualizations that require `nbformat` library. DO NOT include `fig.show()`. The `plot` method must return a Plotly figure object (`fig`). Think step by step."""
            }
            template = f"""
import plotly.express as px
import pandas as pd
import dask.dataframe as dd
<imports>
def plot(data):
    # Check if data is a Dask DataFrame and sample if necessary
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    fig = <stub>  # only modify this section

    return fig

chart = plot(data)  # Data is already loaded. No additional code beyond this line.
"""
        elif library == "datashader":
            instructions = {
        "role": "system",
        "content": f"""{general_instructions}

**Specific Instructions for Datashader:**

- **Use only basic, well-documented functions from Datashader**.
- **Avoid using any functions or methods unless you are certain they exist in Datashader's API**.
- In the `<stub>` section, write code to:
  - Create the plot using basic Datashader functions like `Canvas`, `aggregate`, and `shade`.

**Ensure that:**

- The code is **simple**, **complete**, and **executable**.
- All variables and functions used are defined within the code.
- The `plot(data)` function returns the Datashader image (`img`).

**Do Not Include:**

- Any explanations, comments, or extraneous text.
- Any code to load the data (it's already loaded in `data`).
"""
    }
            template = f"""
                import datashader as ds
                import datashader.transfer_functions as tf
                import pandas as pd
                from colorcet import fire  
                <imports>
                def plot(data):
                    # Data preprocessing
                    <stub>
                    return img

                chart = plot(data)  # Data is already loaded. No additional code beyond this line.
        """
        else:
            raise ValueError(
                "Unsupported library. Choose from 'matplotlib', 'seaborn', 'plotly', 'ggplot', 'altair', and 'datashader'."
            )

        return template, instructions