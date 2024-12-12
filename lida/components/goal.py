import json
import logging
from typing import List, Union
from lida.utils import clean_code_snippet
from lida.datamodel import Goal, TextGenerationConfig, Persona, Summary
from langchain import PromptTemplate
from .textgen_langchain import TextGeneratorLLM  # Use relative import
from dotenv import load_dotenv
from llmx import llm  # Import llm function
from dataclasses import asdict  # Add this import
import pandas as pd
import dask.dataframe as dd  # Add this line

logger = logging.getLogger("lida")
print("Goal loadeggg")
SYSTEM_INSTRUCTIONS = """
You are an experienced data analyst who can generate insightful GOALS about data, including visualization suggestions and rationales. The goals you generate must include the following fields:

- **question**: A clear, concise question that can be answered by visualizing the data.
- **visualization**: A description of the visualization, referencing exact column fields from the summary.
- **rationale**: Justification for the dataset fields used and what will be learned from the visualization.
- **plot_type**: The type of plot to use (e.g., 'scatter', 'line', 'bar').
- **x_axis**: The field(s) to use for the x-axis.
- **y_axis**: The field(s) to use for the y-axis.
- **color**: (Optional) The field to use for color encoding.
- **size**: (Optional) The field to use for size encoding.

Ensure that each goal mentions the exact fields from the dataset summary and considers using data aggregation techniques to avoid overplotting when dealing with large datasets.
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A VALID JSON LIST OF OBJECTS USING THE FOLLOWING FORMAT:

[
    {{
        "index": 0,
        "question": "Your question here",
        "visualization": "Description of the visualization",
        "rationale": "Explanation of why this visualization is useful",
        "plot_type": "Type of plot",
        "x_axis": "Field(s) for x-axis",
        "y_axis": "Field(s) for y-axis",
        "color": "Field for color encoding (optional)",
        "size": "Field for size encoding (optional)"
    }},
    ...
]
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

class GoalExplorer:
    """Generate goals with visualization suggestions and rationales based on a data summary."""

    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge', api_key: str = None):
        """
        Initialize the GoalExplorer with specified model configuration.

        Args:
            model_type (str): Type of the model (e.g., 'cohere').
            model_name (str): Name of the model to use.
            api_key (str): API key for the model provider.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key

        # Initialize the TextGenerator with the specified provider
        self.text_gen = self._initialize_text_generator()

        # Wrap the TextGenerator with TextGeneratorLLM for LangChain compatibility
        self.llm = TextGeneratorLLM(text_gen=self.text_gen, system_prompt=SYSTEM_INSTRUCTIONS)

        # Define the prompt template with input variables
        self.prompt_template = PromptTemplate(
            input_variables=["n", "summary", "persona_description"],
            template=f"""
{SYSTEM_INSTRUCTIONS}

The number of GOALS to generate is {{n}}. The goals should be based on the data summary below:

{{summary}}

The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{{persona_description}}' persona, who is interested in complex, insightful goals about the data.

{FORMAT_INSTRUCTIONS}
"""
        )

        # Update the goal chain initialization
        self.llm_chain = self.prompt_template | self.llm

    def _initialize_text_generator(self):
        """
        Initialize the TextGenerator with the specified provider.

        Returns:
            TextGenerator: An instance of a concrete TextGenerator subclass.
        """
        kwargs = {
            'provider': self.model_type,
            'api_key': self.api_key,
        }

        if self.model_type.lower() == 'cohere':
            kwargs['model'] = self.model_name  # Use 'model' for Cohere
        else:
            kwargs['model_name'] = self.model_name  # Use 'model_name' for other providers

        return llm(**kwargs)

    def generate(
        self,
        summary: Union[dict, Summary],
        textgen_config: TextGenerationConfig,
        n: int = 5,
        persona: Persona = None
    ) -> List[Goal]:
        """
        Generate goals based on a summary of data.

        Args:
            summary (Union[dict, Summary]): Summary of the dataset.
            textgen_config (TextGenerationConfig): Configuration for text generation.
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

        # Convert summary to a dictionary if it's an instance of Summary
        if hasattr(summary, "dict"):
            summary = asdict(summary)
        else:
            summary_dict = summary  # Assume it's already a dict

        if isinstance(summary, dict):
            summary_dict = summary
        elif isinstance(summary, Summary):
            summary_dict = asdict(summary)  # Use asdict for dataclasses
        else:
            raise TypeError("Summary must be a dict or an instance of Summary.")
        # Prepare variables for the prompt
        print("summary_dict", summary_dict)
        prompt_vars = {
            "n": n,
            "summary": json.dumps(summary_dict, indent=4),
            "persona_description": persona_description
        }

        logger.debug(f"Generating goals with variables: {prompt_vars}")

        # Update LLM parameters based on textgen_config
        if textgen_config:
            self.llm.temperature = textgen_config.temperature
            self.llm.max_tokens = textgen_config.max_tokens
            if textgen_config.stop:
                self.llm.stop = textgen_config.stop

        try:
            # Generate the goals using the updated chain
            response = self.llm_chain.invoke(prompt_vars)
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
