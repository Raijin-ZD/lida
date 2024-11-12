import ast
import base64
import importlib
import io
import os
import re
import traceback
from typing import Any, List
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import dask.dataframe as dd
from colorcet import fire
from lida.datamodel import ChartExecutorResponse, Summary
import numpy as np
import ast
def preprocess_code(code: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""

    #code = code.replace("<imports>", "")
    #code = code.replace("<stub>", "")
    #code = code.replace("<transforms>", "")
     # Basic clean-up of placeholder tags
    code = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).strip('```').strip(),code)
    code = code.strip()

    code = code.replace("<imports>", "").replace("<stub>", "").replace("<transforms>", "")

    # Check if the code is empty or invalid type
    if not code or not isinstance(code, str):
        print("Error: Generated code is empty or invalid.")
        return None


    # Syntax validation to catch issues like unclosed brackets or missing colons
    try:
        ast.parse(code)
    except SyntaxError as e:
        print("Syntax error in generated code:", e)
        return None  # Return None to skip invalid code

    # Remove extraneous Markdown-style formatting (e.g., triple backticks)
    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]

    # Ensure that "chart = plot(data)" is in the final version of the code
    # Only keep content up to "chart = plot(data)" if it exists
    if "chart = plot(data)" in code:
        index = code.find("chart = plot(data)")
        code = code[: index + len("chart = plot(data)")]

    # If "chart = plot(data)" is not yet included, append it
    if "chart = plot(data)" not in code:
        code += "\nchart = plot(data)"

    # Final check for the 'import' statement presence and ensure clean imports
    if "import" in code:
        index = code.find("import")
        code = code[index:]

    return code


def get_globals_dict(code_string, data):
    # Ensure code_string is a valid string
    if not code_string or not isinstance(code_string, str):
        print("Error: code_string is invalid.")
        return {}
    
    # Parse the code string into an AST
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        print("Syntax error when parsing code_string:", e)
        print("Code with syntax error:")
        print(code_string)
        return {}
    # Extract the names of the imported modules and their aliases
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = importlib.import_module(alias.name)
                imported_modules.append((alias.name, alias.asname, module))
        elif isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                imported_modules.append(
                    (f"{node.module}.{alias.name}", alias.asname, obj)
                )

    # Import the required modules into a dictionary
    globals_dict = {}
    for module_name, alias, obj in imported_modules:
        if alias:
            globals_dict[alias] = obj
        else:
            globals_dict[module_name.split(".")[-1]] = obj

    ex_dicts = {"pd": pd, "data": data, "plt": plt ,'np' : np , 'ds' : ds ,'tf' : tf,'is_numeric_dtype': pd.api.types.is_numeric_dtype,
        'is_string_dtype': pd.api.types.is_string_dtype,}
    globals_dict.update(ex_dicts)
    return globals_dict


class ChartExecutor:
    """Execute code and return chart object"""

    def __init__(self) -> None:
        pass

    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library="altair",
        return_error: bool = False,
    ) -> Any:
        """Validate and convert code"""

        # # check if user has given permission to execute code. if env variable
        # # LIDA_ALLOW_CODE_EVAL is set to '1'. Else raise exception
        # if os.environ.get("LIDA_ALLOW_CODE_EVAL") != '1':
        #     raise Exception(
        #         "Permission to execute code not granted. Please set the environment variable LIDA_ALLOW_CODE_EVAL to '1' to allow code execution.")

        if isinstance(summary, dict):
            summary = Summary(**summary)

        charts = []
        code_spec_copy = code_specs.copy()
        code_specs = [preprocess_code(code) for code in code_specs]
        if library == "altair":
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    if isinstance(data, dd.DataFrame) or len(data) > 100000:
                        canvas = ds.Canvas(plot_width=800, plot_height=600)
                        agg = canvas.points(data, 'x', 'y', ds.reductions.count())
                        img = tf.shade(agg)
                        img.to_pil()  # Use this for rendering in large data cases
                    else:
                        exec(code, ex_locals)
                        chart = ex_locals["chart"]
                    vega_spec = chart.to_dict()
                    del vega_spec["data"]
                    if "datasets" in vega_spec:
                        del vega_spec["datasets"]

                    vega_spec["data"] = {"url": f"/files/data/{summary.file_name}"}
                    charts.append(
                        ChartExecutorResponse(
                            spec=vega_spec,
                            status=True,
                            raster=None,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code_spec_copy, "\n===========\n")
                    print(exception_error)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts
        elif library == "matplotlib" or library == "seaborn":
            # print colum dtypes
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    # print(ex_locals)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]
                    if plt:
                        buf = io.BytesIO()
                        plt.box(False)
                        plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
                        # try:
                        #     plt.draw()
                        #     # plt.tight_layout()
                        # except AttributeError:
                        #     print("Warning: tight_layout encountered an error. The layout may not be optimal.")
                        #     pass

                        plt.savefig(buf, format="png", dpi=100, pad_inches=0.2)
                        buf.seek(0)
                        plot_data = base64.b64encode(buf.read()).decode("ascii")
                        plt.close()
                    charts.append(
                        ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code_spec_copy[0])
                    print("****\n", str(exception_error))
                    # print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts
        # executor.py

        elif library == "datashader":
            charts = []
            for code in code_specs:
                code = preprocess_code(code)
                if not code:
                    continue  # Skip invalid code

                try:
                    # Prepare the execution environment
                    ex_globals = {
                        'data': data,
                        'ds': ds,
                        'tf': tf,
                        'np': np,
                        'pd': pd,
                        'dd': dd,  # Dask DataFrame support
                    }
                    ex_locals = {}

                    # Execute the generated code
                    exec(code, ex_globals, ex_locals)
                    img = ex_locals.get("chart") or ex_globals.get("chart")

                    if img is None:
                        raise ValueError("The generated code did not produce a 'chart' object.")

                    # Convert the Datashader image to base64
                    buf = io.BytesIO()
                    img.to_pil().save(buf, format="PNG")
                    buf.seek(0)
                    plot_data = base64.b64encode(buf.read()).decode('utf-8')

                    # Collect the result
                    charts.append(
                        ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print("Error in Datashader plot generation:", exception_error)
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts

        elif library == "ggplot":
            # print colum dtypes
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]
                    if plt:
                        buf = io.BytesIO()
                        chart.save(buf, format="png")
                        plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    charts.append(
                        ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts

        elif library == "plotly":
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]

                    if pio:
                        chart_bytes = pio.to_image(chart, 'png')
                        plot_data = base64.b64encode(chart_bytes).decode('utf-8')

                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=True,
                                raster=plot_data,
                                code=code,
                                library=library,
                            )
                        )
                except Exception as exception_error:
                    print(code)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts

        else:
            raise Exception(
                f"Unsupported library. Supportedddd libraries are altair, matplotlib, seaborn, ggplot, plotly. You provided {library}"
            )
