import json
import logging
from typing import List, Union
import pandas as pd
from lida.utils import clean_code_snippet
from lida.datamodel import Goal, TextGenerationConfig, Persona
from langchain.llms import Cohere
from langchain import LLMChain, PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger("lida")

# Define SYSTEM_INSTRUCTIONS and FORMAT_INSTRUCTIONS as class attributes or constants
SYSTEM_INSTRUCTIONS = """
You are an experienced data analyst who can generate insightful GOALS about data, including visualization suggestions and rationales. The visualizations you recommend must follow best practices (e.g., use bar charts instead of pie charts for comparing quantities) and be meaningful (e.g., plot longitude and latitude on maps where appropriate). Each goal must include a question, a visualization (referencing exact column fields from the summary), and a rationale (justification for dataset fields used and what will be learned from the visualization). Each goal must mention the exact fields from the dataset summary above.
Ensure that goals involving visualization of large datasets consider using data aggregation techniques such as histograms, density plots, or other summary-based visualizations to avoid overplotting.
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A VALID JSON LIST OF OBJECTS USING THE FOLLOWING FORMAT:

[
    { 
        "index": 0,  
        "question": "What is the distribution of X?", 
        "visualization": "Histogram of X", 
        "rationale": "This visualization shows the distribution of X to understand its variability."
    },
    ...
]
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

class GoalExplorer:
    """Generate goals with visualization suggestions and rationales based on a data summary."""

    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge-nightly', api_key: str = None):
        """
        Initialize the GoalExplorer with specified model configuration.

        Args:
            model_type (str): Type of the model (e.g., 'cohere').
            model_name (str): Name of the model to use.
            api_key (str): API key for the model provider.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key or os.getenv('COHERE_API_KEY')

        if self.model_type.lower() == 'cohere':
            if not self.api_key:
                raise ValueError("Cohere API key must be provided either via parameter or 'COHERE_API_KEY' environment variable.")
            self.llm = Cohere(
                model=self.model_name,
                cohere_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Define the prompt template with input variables: n, summary, persona
        self.prompt_template = PromptTemplate(
            input_variables=["n", "summary", "persona_description"],
            template=f"""
{SYSTEM_INSTRUCTIONS}

The number of GOALS to generate is {{n}}. The goals should be based on the data summary below:

{{summary}}

The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{{{{persona_description}}}}' persona, who is interested in complex, insightful goals about the data.

{FORMAT_INSTRUCTIONS}
"""
        )

        # Initialize the LLMChain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate(self, summary: dict, textgen_config: TextGenerationConfig, n: int =5, persona: Persona = None) -> List[Goal]:
        """
        Generate goals based on a summary of data.

        Args:
            summary (dict): Summary of the dataset.
            textgen_config (TextGenerationConfig): Configuration for text generation (e.g., max tokens).
            n (int, optional): Number of goals to generate. Defaults to 5.
            persona (Persona, optional): Persona details. Defaults to None.

        Returns:
            List[Goal]: A list of generated goals with visualization suggestions and rationales.
        """
        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can develop complex, insightful goals about data",
                rationale=""
            )

        persona_description = persona.persona

        # Prepare variables for the prompt
        prompt_vars = {
            "n": n,
            "summary": json.dumps(summary, indent=4),
            "persona_description": persona_description
        }

        logger.debug(f"Generating goals with variables: {prompt_vars}")

        try:
            # Generate the goals using LLMChain
            response = self.llm_chain.run(prompt_vars)
            logger.debug(f"Raw response from LLM: {response}")

            # Clean and parse the JSON output
            json_string = clean_code_snippet(response)
            goals_data = json.loads(json_string)

            # Ensure it's a list
            if isinstance(goals_data, dict):
                goals_data = [goals_data]

            # Convert to list of Goal objects
            goals = [Goal(**goal) for goal in goals_data]
            return goals

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            logger.error(f"Response received: {response}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting to generate goals. "
                "Consider using a larger model or a model with higher max token length."
            ) from e
        except Exception as e:
            logger.error(f"An error occurred during goal generation: {e}")
            raise e
