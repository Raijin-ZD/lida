import logging
from typing import Dict, List
from llmx import llm, TextGenerator, TextGenerationConfig
from ..scaffold import ChartScaffold
from lida.datamodel import Goal

logger = logging.getLogger("lida")

system_prompt = """
You are an expert data visualization assistant tasked with generating a complete, executable Python script to create a visualization based on the provided dataset summary and visualization goal.

**Requirements:**
1. Write simple, clear code using only basic, well-documented functions from the specified visualization library.
2. Avoid using advanced or less-known features unless absolutely necessary.
3. Handle both Pandas and Dask DataFrames.
4. Return the visualization object (e.g., `fig`, `chart`, `img`) directly from the `plot` function.
5. Do not include any explanations or extra text.
6. Start with import statements.
"""

class VizGenerator:
    """Generate visualizations from prompt"""

    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge', api_key: str = None):
        self.scaffold = ChartScaffold()
        # Initialize text generator with default settings
        self.text_gen = llm(provider=model_type, model=model_name, api_key=api_key)

    def generate(
        self,
        summary: Dict,
        goal: Goal,
        textgen_config: TextGenerationConfig = None,
        library: str = 'seaborn'
    ) -> List[str]:
        """Generate visualization code given a summary and a goal"""
        try:
            # Get template from scaffold
            template = self.scaffold.get_template(goal, library)
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
Generate visualization code using {library} library.
Dataset summary: {summary}
Template structure:
{template}

Goal: {goal.question}

Generate only the code, no explanations."""}
            ]

            # Use text_gen to generate code with config
            response = self.text_gen.generate(messages=messages, config=textgen_config)
            if response and hasattr(response, 'text'):
                return [msg['content'] for msg in response.text]
            
            return []
            
        except Exception as e:
            logger.error(f"Error generating visualization code: {str(e)}")
            return []
