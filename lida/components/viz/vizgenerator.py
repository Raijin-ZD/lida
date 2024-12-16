# vizgenerator.py
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Union, Any
import re
from dataclasses import asdict
from pydantic import PrivateAttr, Field
from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from llmx import llm
from ..textgen_langchain import TextGeneratorLLM
from lida.utils import clean_code_snippet
from lida.components.scaffold import ChartScaffold
from lida.datamodel import Goal, Summary, TextGenerationConfig
import dask.dataframe as dd
from transformers import GPT2Tokenizer
from pydantic import Field

logger = logging.getLogger("lida")
print("Viz loaded 65bbasfasfx3am")

SYSTEM_INSTRUCTIONS = """
You are an experienced data visualization developer.

CRITICAL RULES:
1. **Use ONLY the existing 'data' variable.**
2. **Do NOT load or create new data.**
3. **When specifying an action, use the exact tool names provided.**
4. **Use the specified visualization library without changing it.**

AVAILABLE TOOLS:
- **code_generator**
- **code_validator**

When taking an action, you must specify the tool name exactly as it appears above, including lowercase letters and underscores.
""".strip()

FORMAT_INSTRUCTIONS = """
RESPONSE FORMAT:
```python
def plot(data):
    # Your visualization code here using only the 'data' variable
    return visualization_object
"""

class CustomOutputParser(AgentOutputParser):
    _tools: List[BaseTool] = PrivateAttr()

    def __init__(self, tools: List[BaseTool]):
        super().__init__()
        self._tools = tools

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        cleaned_output = llm_output.strip()

        # Check for code block first
        if "```python" in cleaned_output:
            code_block = self._extract_code_block(cleaned_output)
            if code_block:
                return AgentFinish(
                    return_values={"output": code_block},
                    log=llm_output
                )

        # Then check for action
        if "Action:" in cleaned_output:
            match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", cleaned_output, re.DOTALL)
            if match:
                action_name = match.group(1).strip().lower().replace(' ', '_')

                # Use self._tools instead of self.tools
                if action_name not in [tool.name for tool in self._tools]:
                    raise OutputParserException(f"Invalid tool name: {action_name}")
                return AgentAction(
                    tool=action_name,
                    tool_input=match.group(2).strip(),
                    log=llm_output
                )

        raise OutputParserException("Invalid output format")

    def _extract_code_block(self, text: str) -> str:
        """Extract valid Python code block"""
        code_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if code_match and "def plot(data):" in code_match.group(1):
            return code_match.group(1).strip()
        return ""

class CodeGenerationTool(BaseTool):
    name: str = Field(default="code_generator")
    description: str = Field(default="Generates visualization code based on data summary and goals")

    def _run(self, inputs: Union[str, dict], **kwargs: Any) -> str:
        try:
            if isinstance(inputs, str):
                # Parse the JSON string
                data = json.loads(inputs)
            elif isinstance(inputs, dict):
                data = inputs
            else:
                return "Invalid input format"

            library = data.get('library', 'seaborn')
            summary = data.get('summary', {})
            goal_dict = data.get('goal', {})
  
            # Ensure goal_dict has all required fields
            goal_dict.setdefault("question", "")
            goal_dict.setdefault("visualization", "")
            goal_dict.setdefault("rationale", "")
            goal_dict.setdefault("plot_type", "scatter")
            goal_dict.setdefault("x_axis", "")
            goal_dict.setdefault("y_axis", "")
            goal_dict.setdefault("color", None)
            goal_dict.setdefault("size", None)
            goal_dict.setdefault("index", 0)
            goal = Goal(**goal_dict)

            scaffold = ChartScaffold()
            template = scaffold.get_template(goal, library)
            return template
        except json.JSONDecodeError as e:
            return f"JSON decode error: {str(e)}"
        except Exception as e:
            return str(e)
        
    def _arun(self, inputs: str) -> str:
        raise NotImplementedError("Async not implemented")
class CodeValidationTool(BaseTool):
    name: str = Field(default="code_validator")
    description: str = Field(default="Validates Python visualization code")
    
    def _run(self, code: str) -> str:
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            
            # Check required elements
            if "def plot(data):" not in code:
                return "Missing plot function definition"
            if "return" not in code:
                return "Missing return statement"
                
            return "Code validation passed"
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def _arun(self, code: str) -> str:
        raise NotImplementedError("Async not implemented")

class VizGenerator:
    def __init__(self, data=None, model_type: str = 'cohere', model_name: str = 'command-xlarge', api_key: str = None):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Initialize tokenizer

        self.text_gen = self._initialize_text_generator()
        
        # Initialize with validated config
        self.default_config = TextGenerationConfig(
            temperature=0.1,
            max_tokens=2000,  # Add reasonable default
            stop=None
        )
        
        self.llm = TextGeneratorLLM(
            text_gen=self.text_gen, 
            system_prompt=SYSTEM_INSTRUCTIONS,
            config=self.default_config,
            temperature=self.default_config.temperature,
            max_tokens=self.default_config.max_tokens
        )
        
        # Single validation tool
        self.tools = [
            CodeGenerationTool(),
            CodeValidationTool(name="code_validator")
        ]
        
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="history", k=0, return_messages=True)

        self.prompt_template = PromptTemplate(
    input_variables=["input"],
    template="""
You are an experienced data visualization developer.

Use ONLY the existing 'data' variable.
Do NOT load or create new data.
Use the specified visualization library: {input['library']}.

DATASET SUMMARY:
{input['summary']}

VISUALIZATION GOAL:
{input['goal']}

Generate the visualization code now.

def plot(data):
    # Your visualization code here
    return visualization_object
""".strip()
)

        self.viz_chain = self.prompt_template | self.llm

        # Updated agent configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            early_stopping_method="force",  # Changed from "generate" to "force"
            agent_kwargs={
                "output_parser": CustomOutputParser(self.tools),
                "prompt": self.prompt_template,
            }
        )

    def _initialize_text_generator(self):
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
        try:
            config = textgen_config or self.default_config
            self._update_llm_config(config)
            
            # Serialize 'input' to JSON string
            agent_input = json.dumps({
                "summary": self._prepare_summary(summary),
                "goal": self._prepare_goal(goal),
                "library": library,
            })

            response = self.agent.run(agent_input)
            
            code = self._extract_code(response)
            if not code:
                raise ValueError("No valid code generated")
                
            validation = self.tools[1]._run(code)
            if "passed" not in validation.lower():
                raise ValueError(f"Code validation failed: {validation}")
                
            return [code]
            
        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}", exc_info=True)
            return []

    def _extract_code(self, response: str) -> str:
        """Extract and validate code from response"""
        # Updated regex to match code blocks with or without 'python'
        code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            if "def plot(data):" in code and "return" in code:
                try:
                    # Basic syntax check
                    compile(code, '<string>', 'exec')
                    # Clean the code to remove any code block markers
                    code = clean_code_snippet(code)
                    return code
                except SyntaxError:
                    logger.error("Generated code has a syntax error.")
                    return ""
        return ""

    def _prepare_summary(self, summary):
        if isinstance(summary, dict):
            return summary
        elif hasattr(summary, 'dict'):
            return summary.dict()
        elif hasattr(summary, '_asdict'):
            return asdict(summary)
        return summary

    def _prepare_goal(self, goal):
        if isinstance(goal, dict):
            goal_dict = goal
        elif hasattr(goal, 'dict'):
            goal_dict = goal.dict()
        elif hasattr(goal, '_asdict'):
            goal_dict = asdict(goal)
        else:
            goal_dict = {"visualization": str(goal)}
        if 'visualization' not in goal_dict or not goal_dict['visualization']:
            goal_dict['visualization'] = goal_dict.get('question', '')
        return goal_dict

    def _update_llm_config(self, textgen_config):
        self.llm.temperature = textgen_config.temperature
        self.llm.max_tokens = textgen_config.max_tokens
        if textgen_config.stop:
            self.llm.stop = textgen_config.stop
