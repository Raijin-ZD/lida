import ast
import base64
import importlib
import io
import os
import re
import traceback
import logging
from typing import Any, List, Union, Dict
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import dask.dataframe as dd
from colorcet import fire
from lida.datamodel import ChartExecutorResponse, Summary, Goal
import numpy as np
from .viz.vizgenerator import VizGenerator
from .agent.code_repair_agent import CodeRepairAgent
from llmx import llm
from lida.datamodel import TextGenerationConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

def preprocess_code(code: str) -> str:
    """Clean and validate code string"""
    if not code or not isinstance(code, str):
        return ""
    
    # Remove markdown code block syntax if present
    if "```" in code:
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            code = match.group(1)
    
    code = str(code).strip()
    
    # Validate basic structure
    if "def plot(data):" not in code:
        return ""
        
    return code

def get_globals_dict(code_string, data):
    """Set up execution environment with necessary modules"""
    if not code_string or not isinstance(code_string, str):
        return {}
    
    globals_dict = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'ds': ds,
        'tf': tf,
        'dd': dd,
        'data': data,
        'is_numeric_dtype': pd.api.types.is_numeric_dtype,
        'is_string_dtype': pd.api.types.is_string_dtype,
    }
    
    # Add any additional modules from imports in code
    try:
        tree = ast.parse(code_string)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        module = importlib.import_module(alias.name)
                        globals_dict[alias.asname or alias.name] = module
                    except ImportError:
                        continue
    except SyntaxError:
        pass
        
    return globals_dict

class ChartExecutor:
    """Execute code and return chart object"""

    def __init__(self):
        self.viz_generator = VizGenerator()
        self.code_repair_agent = CodeRepairAgent(
            text_gen=llm,
            textgen_config=TextGenerationConfig()
        )
        self.debug = True  # Add debug flag

    def execute(
        self,
        code_specs: Union[List[str], str],
        data: Any,
        summary: Union[dict, Summary],
        library: str = "seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        """
        Execute visualization code and return chart response.

        Args:
            code_specs: Code to execute (string or list of strings)
            data: Data to visualize
            summary: Dataset summary
            library: Visualization library to use
            return_error: Whether to return error details
        """
        try:
            # Normalize code_specs to list
            if isinstance(code_specs, str):
                code_specs = [code_specs]
            elif isinstance(code_specs, list) and len(code_specs) > 0:
                if isinstance(code_specs[0], list):
                    code_specs = code_specs[0]
                elif isinstance(code_specs[0], dict) and 'content' in code_specs[0]:
                    code_specs = [spec['content'] for spec in code_specs]

            # Ensure summary is proper type
            if isinstance(summary, dict):
                summary = Summary(**summary)

            charts = []
            
            # Process each code spec
            for code in code_specs:
                try:
                    # Clean and validate code
                    processed_code = preprocess_code(code)
                    if not processed_code:
                        continue

                    logger.info(f"Processing code for {library}:")
                    logger.info(f"Original code:\n{code}")

                    # Special handling for datashader library
                    if library == "datashader":
                        # Keep data as Dask DataFrame if it is one
                        data_for_execution = data
                        
                        # Modify code to handle array comparisons
                        processed_code = processed_code.replace(
                            "if isinstance(data, dd.DataFrame):",
                            "if hasattr(data, 'compute'):"
                        )
                        
                        # Set up minimal globals for datashader
                        globals_dict = {
                            '__builtins__': __builtins__,
                            'data': data_for_execution,
                            'ds': ds,
                            'tf': tf,
                            'np': np,
                            'pd': pd,
                            'dd': dd,
                            'fire': fire
                        }

                        # Execute with numpy error handling
                        with np.errstate(all='ignore'):
                            exec(processed_code, globals_dict)
                            
                        img = globals_dict.get("chart")
                        if img is None:
                            raise ValueError("No chart object was created")

                        # Convert to PNG
                        buf = io.BytesIO()
                        img.to_pil().save(buf, format='PNG')
                        buf.seek(0)
                        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                        
                        charts.append(ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=processed_code,
                            library=library,
                        ))
                        
                    else:
                        # Existing handling for other libraries
                        # Try to repair code first
                        try:
                            logger.info("Attempting code repair...")
                            repaired_code = self.code_repair_agent.repair(processed_code)
                            if repaired_code != processed_code:
                                logger.info("Code was repaired!")
                                logger.info(f"Repaired code:\n{repaired_code}")
                                processed_code = repaired_code
                            else:
                                logger.info("No repairs needed")
                        except Exception as repair_error:
                            logger.warning(f"Code repair failed: {repair_error}")
                            # Continue with original code if repair fails

                        # Prepare data
                        if isinstance(data, dd.DataFrame):
                            data_for_execution = data.compute() if library != "datashader" else data
                        else:
                            data_for_execution = data

                        # Set up execution environment
                        globals_dict = get_globals_dict(processed_code, data_for_execution)
                        
                        # Execute code with error handling for numpy comparisons
                        try:
                            with np.errstate(all='ignore'):  # Suppress numpy warnings
                                exec(processed_code, globals_dict)
                        except ValueError as ve:
                            if "truth value of an array" in str(ve):
                                # Modify code to handle array comparisons
                                processed_code = processed_code.replace(" == ", ".equals(")
                                processed_code = processed_code.replace(" != ", ".ne(")
                                exec(processed_code, globals_dict)
                        
                        chart = globals_dict.get("chart") or globals_dict.get("fig")
                        if chart is None:
                            raise ValueError("No chart object was created")

                        # Convert chart based on library
                        if library in ["matplotlib", "seaborn"]:
                            buf = io.BytesIO()
                            plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            plot_data = base64.b64encode(buf.read()).decode("ascii")
                            plt.close()
                            charts.append(ChartExecutorResponse(
                                spec=None, status=True, raster=plot_data,
                                code=processed_code, library=library
                            ))
                        elif library == "plotly":
                            chart_bytes = pio.to_image(chart, format='png')
                            plot_data = base64.b64encode(chart_bytes).decode('utf-8')
                            charts.append(ChartExecutorResponse(
                                spec=None, status=True, raster=plot_data,
                                code=processed_code, library=library
                            ))
                        elif library == "altair":
                            vega_spec = chart.to_dict()
                            if "data" in vega_spec:
                                del vega_spec["data"]
                            if "datasets" in vega_spec:
                                del vega_spec["datasets"]
                            vega_spec["data"] = {"url": f"/files/data/{summary.file_name}"}
                            charts.append(ChartExecutorResponse(
                                spec=vega_spec, status=True, raster=None,
                                code=processed_code, library=library
                            ))
                        elif library == "datashader":
                            buf = io.BytesIO()
                            chart.to_pil().save(buf, format='PNG')
                            buf.seek(0)
                            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                            charts.append(ChartExecutorResponse(
                                spec=None, status=True, raster=plot_data,
                                code=processed_code, library=library
                            ))

                except Exception as e:
                    logger.error(f"Error executing code: {str(e)}")
                    logger.error(f"Code that caused error:\n{processed_code}")
                    if return_error:
                        charts.append(ChartExecutorResponse(
                            spec=None,
                            status=False,
                            raster=None,
                            code=processed_code,
                            library=library,
                            error={
                                "message": str(e),
                                "traceback": traceback.format_exc()
                            }
                        ))

            return charts

        except Exception as e:
            logger.error(f"Error in execute: {str(e)}")
            if return_error:
                return [ChartExecutorResponse(
                    spec=None,
                    status=False,
                    raster=None,
                    code=str(code_specs),
                    library=library,
                    error={"message": str(e), "traceback": traceback.format_exc()}
                )]
            return []

