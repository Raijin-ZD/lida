from dataclasses import asdict
from lida.datamodel import Goal
print("Generating visualization code scaffold...")

class ChartScaffold(object):
    """Return code scaffold for charts in multiple visualization libraries"""

    def __init__(self) -> None:
        pass

    def get_template(self, goal: Goal, library: str):
        plot_type = goal.plot_type
        x_axis = goal.x_axis
        y_axis = goal.y_axis
        color = goal.color
        size = goal.size
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
- If `data` is a Dask DataFrame and the library requires a Pandas DataFrame, convert it (e.g., using `data.compute()`).

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
            template = f"""
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    # Insert plotting code below. No placeholders. Example:
    # plt.plot(data['{x_axis}'], data['{y_axis}'])
    # plt.xlabel('{x_axis}')
    # plt.ylabel('{y_axis}')
    # plt.title('{goal.question}', wrap=True)
    return plt
    
#always type the below line at the end of the code
chart = plot(data)
"""
        elif library == "seaborn":
            template = f"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    # Insert plotting code below. No placeholders. Example:
    # sns.scatterplot(data=data, x='{x_axis}', y='{y_axis}')
    # plt.xlabel('{x_axis}')
    # plt.ylabel('{y_axis}')
    # plt.title('{goal.question}', wrap=True)
    return plt

#always type the below line at the end of the code
chart = plot(data)
"""
        elif library == "ggplot":
            template = f"""
import plotnine as p9
import pandas as pd
import dask.dataframe as dd

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    # Insert plotting code below. Example:
    # chart = (p9.ggplot(data, p9.aes(x='{x_axis}', y='{y_axis}')) + p9.geom_point())
    return chart

#always type the below line at the end of the code
chart = plot(data)
"""
        elif library == "altair":
            template = f"""
import altair as alt
import pandas as pd
import dask.dataframe as dd

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    # Insert plotting code below. Example:
    # chart = alt.Chart(data).mark_point().encode(
    #     x='{x_axis}:Q', 
    #     y='{y_axis}:Q'
    # )
    return chart

#always type the below line at the end of the code
chart = plot(data)
"""
        elif library == "plotly":
            template = f"""
import plotly.express as px
import pandas as pd
import dask.dataframe as dd

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.sample(frac=0.1, random_state=42).compute()
    # Insert plotting code below. Example:
    # fig = px.scatter(data, x='{x_axis}', y='{y_axis}')
    return fig
    
#always type the below line at the end of the code
chart = plot(data)
"""
        elif library == "datashader":
            template = f"""
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import dask.dataframe as dd
from colorcet import fire  # Optional color map

def plot(data):
    if isinstance(data, dd.DataFrame):
        data = data.compute()
    # Insert plotting code below. Example:
    canvas = ds.Canvas(plot_width=800, plot_height=600)
    agg = canvas.points(data, '{x_axis}', '{y_axis}')
    img = tf.shade(agg, cmap=fire)
    return img

#always type the below line at the end of the code
chart = plot(data) 
"""
            return template
        else:
            raise ValueError(
                "Unsupported library. Choose from 'matplotlib', 'seaborn', 'plotly', 'ggplot', 'altair', and 'datashader'."
            )


        return template 

