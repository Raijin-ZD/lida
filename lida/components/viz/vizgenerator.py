# vizgenerator.py
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import asdict

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

logger = logging.getLogger("lida")
print("Viz loaded 65xx3am")

SYSTEM_INSTRUCTIONS = """
You are an experienced data visualization developer. 
Generate code using the specified visualization library. 
Use the provided 'data' directly and return complete, executable code with proper imports.
Do not add explanations or comments outside of the code. Do not rename plot(data).
Make sure the final code snippet is complete and executable.""".strip()

FORMAT_INSTRUCTIONS = """
RESPONSE FORMAT:
```python
# Your code here
"""

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        import re
        action_regex = r"Action\s*:\s*(.*?)\nAction\s*Input\s*:\s*(.*)"
        match = re.search(action_regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        raise OutputParserException(f"Could not parse LLM output: {llm_output}")

class CodeGenerationTool(BaseTool):
    name: str = "code_generator"
    description: str = "Generates visualization code based on data summary and goals"

    def _run(self, inputs: str, **kwargs) -> str:
        try:
            data = json.loads(inputs)
            library = data.get('library', 'datashader')
            summary = data.get('summary', {})
            goal_dict = data.get('goal', {})

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
        except Exception as e:
            return str(e)

class VizGenerator:
    def __init__(self, data=None, model_type: str = 'cohere', model_name: str = 'command-xlarge', api_key: str = None):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Initialize tokenizer

        self.text_gen = self._initialize_text_generator()
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=SYSTEM_INSTRUCTIONS)
        self.tools = [CodeGenerationTool()]
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="history", k=1, return_messages=True)

        self.prompt_template = PromptTemplate(
    input_variables=["summary", "goal", "library", "history"],
    template=f"""
{SYSTEM_INSTRUCTIONS}

ADDITIONAL RULES:

1) Do not rename the plot(data) function.
2)Insert all necessary code (imports, plotting logic) directly into the code snippet.
3) The code returned must be complete, executable, and contain no extra explanations or placeholders.
4) Only use the {{library}} library for the visualization, do not switch to another library.
5) Do not include any explanations or comments outside of the code.
6) dont create any dataframes, use the data variable directly as its already loaded into 'data'.

DATASET SUMMARY: {{summary}}

VISUALIZATION GOAL: {{goal}}

The data is already loaded into 'data'.

Question: Please generate the visualization code now.

{FORMAT_INSTRUCTIONS}

{{history}} """.strip() )

        self.viz_chain = self.prompt_template | self.llm

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            #memory=self.memory,
            max_iterations=1,
            handle_parsing_errors=True,
            agent_kwargs={
                "output_parser": CustomOutputParser(),
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
        if hasattr(summary, "to_dict"):
            summary_dict = summary.to_dict()
        else:
            summary_dict = summary

        if hasattr(goal, "to_dict"):
            goal_dict = goal.to_dict()
        else:
            goal_dict = goal

        summary_dict = self._prepare_summary(summary)
        goal_dict = self._prepare_goal(goal)

        if 'visualization' not in goal_dict or not goal_dict['visualization']:
            goal_dict['visualization'] = goal_dict.get('question', '')

        agent_input = {
            "summary": summary_dict,
            "goal": goal_dict,
            "library": library,
        }
        
        print("Agent Input JSON:", json.dumps(agent_input, indent=2))

        if textgen_config:
            self._update_llm_config(textgen_config)

        if data is not None:
            self.data = data

        if self.data is not None:
            if isinstance(self.data, dd.DataFrame) and library != 'datashader':
                self.data = self.data.sample(frac=0.1).compute()
        else:
            raise ValueError("Data must be provided for visualization generation.")

        memory_variables = self.memory.load_memory_variables({})
        history = memory_variables.get("history", "")

        prompt = self.prompt_template.format(
            summary=json.dumps(summary_dict),
            goal=json.dumps(goal_dict),
            library=library,
            history=history
        )

        # Truncate prompt if it exceeds the token limit
        #max_tokens = 4081
        #tokens = self.tokenizer.encode(prompt)
        #if len(tokens) > max_tokens:
           # prompt = self.tokenizer.decode(tokens[:max_tokens])

        try:
            response = self.agent.run(json.dumps(agent_input))
            code = clean_code_snippet(response)
            return [code] if code else []
        except Exception as e:
            logger.error(f"Error generating visualization code: {e}")
            raise

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
