# vizgenerator.py
import json
import logging
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

print("Viz loaded 3am")

logger = logging.getLogger("lida")
SYSTEM_INSTRUCTIONS = """
You are an experienced data visualization developer who can generate code based on data summaries and goals.
You must follow these rules:
1. Use only basic, documented functions from the specified visualization library
2. Handle both Pandas and Dask DataFrames appropriately 
3. Return complete, executable code with proper imports
4. Include necessary data preprocessing steps
5. Use appropriate visualization types based on the data and goal
"""

FORMAT_INSTRUCTIONS = """
RETURN ONLY THE VISUALIZATION CODE AS A STRING, WITH NO ADDITIONAL TEXT OR FORMATTING.
THE CODE SHOULD BE COMPLETE AND EXECUTABLE.
"""


class CodeGenerationTool(BaseTool):
    name :str = "code_generator"
    description :str = "Generates visualization code based on data summary and goals"
    
    def _run(self, inputs: str) -> str:
        # Parse the inputs from JSON string
        try:
            data = json.loads(inputs)
            library = data.get('library', 'seaborn')
            summary = data.get('summary', {})
            goal = data.get('goal', {})

            scaffold = ChartScaffold()
            template, instructions = scaffold.get_template(goal, library)
            
            return template
        except Exception as e:
            return str(e)

class DataAnalysisTool(BaseTool):
    name :str = "data_analyzer"
    description :str = "Analyzes data properties to suggest appropriate visualization approaches"
    
    def _run(self, inputs: str) -> str:
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
    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge-nightly', api_key: str = None):
        """Initialize VizGenerator with model configuration"""
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key

        # Initialize text generator
        self.text_gen = self._initialize_text_generator()
        
        # Initialize TextGeneratorLLM
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=SYSTEM_INSTRUCTIONS)

        # Initialize new tools
        self.tools = [
            CodeGenerationTool(),
            DataAnalysisTool(),
            Tool(
                name="template_selector",
                func=self._select_visualization_template,
                description="Selects appropriate visualization template based on data and goal"
            )
        ]

        # Initialize the agent with improved configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
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

    def generate(self, summary, goal, library='seaborn', textgen_config=None):
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
        agent_input = {
            "summary": summary_dict,
            "goal": goal_dict,
            "library": library
        }

        # Update LLM parameters if textgen_config provided
        if textgen_config:
            self._update_llm_config(textgen_config)

        try:
            # Get visualization suggestions from the data analyzer
            analysis = self.tools[1]._run(json.dumps(agent_input))
            analysis_dict = json.loads(analysis)

            # Update library if needed for large datasets
            if analysis_dict.get("large_data") and library != "datashader":
                logger.warning("Large dataset detected, switching to datashader")
                library = "datashader"
            
            # Generate code using the agent
            response = self.agent.run(json.dumps({**agent_input, "analysis": analysis_dict}))
            
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
