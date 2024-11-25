# vizgenerator.py
from typing import Dict
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from dataclasses import asdict
from lida.datamodel import Goal, Summary
from langchain.llms import Cohere

class VizGeneratorTool(BaseTool):
    name: str = "visualization_generator"
    description: str = "Generates visualization code based on data summary and goals"
    
    def _run(self, data_summary: str, goal: str, library: str) -> str:
        # Tool implementation for generating visualization code
        return f"Generated visualization code for {goal} using {library}"

class VizGenerator:
    def __init__(self):
        self.llm = Cohere(
            cohere_api_key="your-api-key",
            model="command",
            temperature=0
        )
        
        self.viz_prompt = PromptTemplate(
            input_variables=["summary", "goal", "library"],
            template="""
            Generate visualization code using {library} based on:
            Dataset Summary: {summary}
            Visualization Goal: {goal}
            
            Follow these rules:
            1. Use only basic, documented functions
            2. Handle both Pandas and Dask DataFrames
            3. Return complete, executable code
            4. Include proper data preprocessing
            """
        )
        
        tools = [VizGeneratorTool()]
        self.agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def generate(self, summary, goal, library='seaborn', textgen_config=None, text_gen=None):
        if isinstance(summary, dict):
            summary = Summary(**summary)
        if isinstance(goal, dict):
            goal = Goal(**goal)
        
        viz_chain = LLMChain(
            llm=self.llm,
            prompt=self.viz_prompt
        )
        
        result = viz_chain.run(
            summary=str(summary),
            goal=goal.question,
            library=library
        )
        
        return [result]
