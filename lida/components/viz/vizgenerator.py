# vizgenerator.py
import json
import logging
import pandas as pd
from typing import List, Dict, Optional
from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from dataclasses import asdict
from lida.datamodel import Goal, Summary, TextGenerationConfig
from ..textgen_langchain import TextGeneratorLLM
from llmx import llm
from lida.utils import clean_code_snippet
from lida.components.summarizer import Summarizer
from lida.components.scaffold import ChartScaffold
from lida.components.goal import GoalExplorer
import dask.dataframe as dd
from types import SimpleNamespace

print("Viz loaded 3am")

logger = logging.getLogger("lida")
SYSTEM_INSTRUCTIONS = """
You are an experienced data visualization developer who can generate code based on data summaries and goals.
You must follow these rules:
1. Use only basic, documented functions from the specified visualization library.
2. Use the provided 'data' variable directly; do not load or read data from files.
3. Return complete, executable code with proper imports.
4. Include necessary data preprocessing steps.
5. Use appropriate visualization types based on the data and goal.
"""

FORMAT_INSTRUCTIONS = """
RETURN ONLY THE VISUALIZATION CODE AS A STRING, WITH NO ADDITIONAL TEXT OR FORMATTING.
THE CODE SHOULD BE COMPLETE AND EXECUTABLE.
"""


class CodeGenerationTool(BaseTool):
    name: str = "code_generator"
    description: str = "Generates visualization code based on data summary and goals"

    def _run(self, inputs: str, **kwargs) -> str:
        try:
            data = json.loads(inputs)
            library = data.get('library', 'seaborn')
            summary = data.get('summary', {})
            goal_dict = data.get('goal', {})

            # Convert 'goal' dict to an object with attributes
            goal = SimpleNamespace(**goal_dict)

            scaffold = ChartScaffold()
            template, instructions = scaffold.get_template(goal, library)

            return template
        except Exception as e:
            return str(e)

class DataAnalysisTool(BaseTool):
    name :str = "data_analyzer"
    description :str = "Analyzes data properties to suggest appropriate visualization approaches"
    
    def _run(self, inputs: str, **kwargs) -> str:
        try:
            data = json.loads(inputs)
            summary = data.get('summary', {})
            # Removed the call to summarizer.summarize(data)
            
            # Analyze column types and suggest visualizations
            suggestions = {
                "large_data": summary.get("rows", 0) > 10000,
                "column_types": summary.get("column_types", {}),
                "suggested_library": "datashader" if summary.get("rows", 0) > 100000 else "seaborn"
            }
            return json.dumps(suggestions)
        except Exception as e:
            return str(e)

class VizGenerator:
    def __init__(self, data=None, model_type: str = 'cohere', model_name: str = 'command-xlarge-nightly', api_key: str = None):
        """Initialize VizGenerator with model configuration"""
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.data = data  # Store the data provided

        # Initialize text generator
        self.text_gen = self._initialize_text_generator()
        
        # Initialize TextGeneratorLLM
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=SYSTEM_INSTRUCTIONS)
        
        # Remove the creation of the Pandas DataFrame Agent
        # Previously we had:
        # self.dataframe_agent = create_pandas_dataframe_agent(
        #     llm=self.llm,
        #     df=self.data if self.data is not None else pd.DataFrame(),
        #     verbose=True,
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        # )

        # Update tools to include only necessary tools
        self.tools = [
            CodeGenerationTool(),
            DataAnalysisTool(),
            Tool(
                name="template_selector",
                func=self._select_visualization_template,
                description="Selects appropriate visualization template based on data and goal"
            ),
            # Remove DataFrame Agent tools
        ]

        # Initialize the agent with updated configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Changed agent type
            verbose=True,
            max_iterations=30  # Increase the iteration limit
        )

        # Define visualization prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["summary", "goal", "library"],
            template=f"""
{SYSTEM_INSTRUCTIONS}

DATASET SUMMARY:
{{summary}}

VISUALIZATION GOAL:
{{goal}}

THE DATA IS ALREADY LOADED INTO A VARIABLE NAMED 'data'.

GENERATE VISUALIZATION CODE USING THE {{library}} LIBRARY.

{FORMAT_INSTRUCTIONS}
"""
        )

        # Create the generation chain
        self.viz_chain = self.prompt_template | self.llm

    def _initialize_text_generator(self):
        """Initialize TextGenerator with provider configuration"""
        kwargs = {
            'provider': self.model_type,
            'api_key': self.api_key,
        }
        
        if self.model_type.lower() == 'cohere':
            kwargs['model'] = self.model_name
        else:
            kwargs['model_name'] = self.model_name

        return llm(**kwargs)

    def generate(self, summary, goal, library='seaborn', data=None, textgen_config=None):
        """Generate visualization code based on summary and goal"""
        # Convert summary to a dictionary if it's an instance of Summary
        if hasattr(summary, "to_dict"):
            summary_dict = summary.to_dict()
        else:
            summary_dict = summary  # Assume it's already a dict

        # Convert goal to a dictionary if it's an instance of Goal
        if hasattr(goal, "to_dict"):
            goal_dict = goal.to_dict()
        else:
            goal_dict = goal  # Assume it's already a dict

        # Prepare the input for the agent
        summary_dict = self._prepare_summary(summary)
        goal_dict = self._prepare_goal(goal)
        # Ensure 'visualization' key exists in goal_dict
        if 'visualization' not in goal_dict or not goal_dict['visualization']:
            goal_dict['visualization'] = goal_dict.get('question', '')
        agent_input = {
            "summary": summary_dict,
            "goal": goal_dict,
            "library": library,
            "data_info": "The data is already loaded into a variable named 'data'."
        }

        # Update LLM parameters if textgen_config provided
        if textgen_config:
            self._update_llm_config(textgen_config)

        # Update data if provided
        if data is not None:
            self.data = data

        # Ensure data is available
        if self.data is not None:
            if isinstance(self.data, dd.DataFrame):
                # If data is a Dask DataFrame, compute or sample
                self.data = self.data.sample(frac=0.1).compute()
            else:
                self.data = self.data
        else:
            raise ValueError("Data must be provided for visualization generation.")

        try:
            # Get visualization suggestions from the data analyzer
            analysis = self.tools[1]._run(json.dumps(agent_input))
            analysis_dict = json.loads(analysis)

            # Update library if needed for large datasets
            if analysis_dict.get("large_data") and library != "datashader":
                logger.warning("Large dataset detected, switching to datashader")
                library = "datashader"
            
            # Update agent input with analysis
            agent_input.update({"analysis": analysis_dict})

            # Generate code using the agent
            response = self.agent.run(json.dumps(agent_input))

            # Clean and return the code
            code = clean_code_snippet(response)
            return [code] if code else []

        except Exception as e:
            logger.error(f"Error generating visualization code: {e}")
            raise

    def _prepare_summary(self, summary):
        """Prepare summary for JSON serialization"""
        if isinstance(summary, dict):
            return summary
        elif hasattr(summary, 'dict'):
            return summary.dict()
        elif hasattr(summary, '_asdict'):
            return asdict(summary)
        return summary

    def _prepare_goal(self, goal):
        """Prepare goal for JSON serialization"""
        if isinstance(goal, dict):
            return goal
        elif hasattr(goal, 'dict'):
            return goal.dict()
        elif hasattr(goal, '_asdict'):
            return asdict(goal)
        return {"question": str(goal)}

    def _update_llm_config(self, textgen_config):
        """Update LLM configuration"""
        self.llm.temperature = textgen_config.temperature
        self.llm.max_tokens = textgen_config.max_tokens
        if textgen_config.stop:
            self.llm.stop = textgen_config.stop

    def _select_visualization_template(self, goal_type: str) -> str:
        """Selects appropriate visualization template based on goal type"""
        templates = {
            "distribution": "sns.histplot(data=data, x='{col}', kde=True)",
            "correlation": "sns.scatterplot(data=data, x='{col1}', y='{col2}')",
            "comparison": "sns.barplot(data=data, x='{col1}', y='{col2}')",
            "trend": "sns.lineplot(data=data, x='{col1}', y='{col2}')",
            "composition": "data.plot.pie(y='{col}')",
        }
        return templates.get(goal_type, "sns.histplot(data=data, x='{col}')")
