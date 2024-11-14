from dataclasses import asdict

from lida.datamodel import Goal

class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""
    print("scaffold tmam")

    def __init__(self) -> None:
        pass

    def get_template(self, goal: Goal, library: str):

        general_instructions = f"""
If the solution requires a single value (e.g., max, min, median, first, last, etc.), ALWAYS add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F). If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column:
i) Convert date fields to date types using data['<field>'] = pd.to_datetime(data['<field>'], errors='coerce'), ALWAYS use errors='coerce'
ii) Drop the rows with NaT values: data = data[pd.notna(data['<field>'])]
iii) Convert field to the right time format for plotting.
ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed).

Solve the task carefully by completing ONLY the `<imports>` AND `<stub>` section.

Given the dataset summary, the `plot(data)` method should generate a {library} chart ({goal.visualization}) that addresses this goal: {goal.question}.

DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable `data`.

The code should handle both Pandas and Dask DataFrames:
- If `data` is a Dask DataFrame and the library requires a Pandas DataFrame, sample a fraction and compute it to obtain a Pandas DataFrame.
- Use appropriate methods for data preprocessing.
"""

        matplotlib_instructions = f"""{general_instructions}
DO NOT include `plt.show()`. The `plot` method must return a matplotlib object (`plt`). Think step by step.
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
                "content": f"{matplotlib_instructions} Use Basemap for charts that require a map."
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
Fill in the `<imports>` section with necessary imports.

Fill in the `<stub>` section with code that:
- If necessary, sample the data if it's too large.
- Preprocess the data using methods compatible with Dask DataFrames.
- Create the plot using Datashader functions.
- Return the Datashader image (`img`) for rendering.

Ensure that:
- The code is complete and executable without missing parts.
- The `plot` function returns the Datashader image (`img`).

Do not include any explanations or extraneous text outside of the code.
"""
            }
            template = f"""
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import dask.dataframe as dd
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