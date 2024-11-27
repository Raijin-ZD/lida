import json
import logging
from typing import List, Union
from lida.utils import clean_code_snippet
from lida.datamodel import Goal, TextGenerationConfig, Persona
from langchain import LLMChain, PromptTemplate
from .textgen_langchain import TextGeneratorLLM  # Use relative import
from dotenv import load_dotenv
from llmx import TextGenerator  # Import TextGenerator from llmx
from llmx import llm  # Import llm function
logger = logging.getLogger("lida")
print("Goal loadedffff")
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
        self.api_key = api_key

        # Initialize the TextGenerator with the specified provider
        self.text_gen = self._initialize_text_generator()

        # Wrap the TextGenerator with TextGeneratorLLM for LangChain compatibility
        self.llm = TextGeneratorLLM(text_gen=self.text_gen)

        # Define the prompt template with input variables
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

    def generate(self, summary: dict, textgen_config: TextGenerationConfig, n: int = 5, persona: Persona = None) -> List[Goal]:
        """
        Generate goals based on a summary of data.

        Args:
            summary (dict): Summary of the dataset.
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

        # Prepare variables for the prompt
        prompt_vars = {
            "n": n,
            "summary": json.dumps(summary, indent=4),
            "persona_description": persona_description
        }

        logger.debug(f"Generating goals with variables: {prompt_vars}")

        # Update LLM parameters based on textgen_config
        if textgen_config:
            self.llm.temperature = textgen_config.temperature
            self.llm.max_tokens = textgen_config.max_new_tokens
            if textgen_config.stop:
                self.llm.stop = textgen_config.stop

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
