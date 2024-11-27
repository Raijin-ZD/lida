# manager.py

import os
from typing import List, Union
import logging
import pandas as pd
from lida.datamodel import Goal, Summary, TextGenerationConfig, Persona
from lida.utils import read_dataframe
from .summarizer import Summarizer  # Ensure correct import
from .goal import GoalExplorer
from .persona import PersonaExplorer
from .executor import ChartExecutor
from .viz import VizGenerator, VizEditor, VizExplainer, VizEvaluator, VizRepairer, VizRecommender

import lida.web as lida

logger = logging.getLogger("lida")

print("manager.py is being importedggg.")  # to see if it's updated

class Manager:
    def __init__(self, model_type: str = 'cohere', model_name: str = 'command-xlarge-nightly', api_key: str = None, **kwargs):
        """
        Initialize the Manager with specified model configuration and other components.

        Args:
            model_type (str): Type of the model (e.g., 'cohere').
            model_name (str): Name of the model to use.
            api_key (str): API key for the model provider.
            **kwargs: Additional keyword arguments.
        """
        # Initialize the Summarizer with provided configuration
        self.summarizer = Summarizer(model_type=model_type, model_name=model_name, api_key=api_key)

        # Initialize GoalExplorer with provided configuration
        self.goal = GoalExplorer(model_type=model_type, model_name=model_name, api_key=api_key)

        # Initialize other components as before
        self.vizgen = VizGenerator()
        self.vizeditor = VizEditor()
        self.executor = ChartExecutor()
        self.explainer = VizExplainer()
        self.evaluator = VizEvaluator()
        self.repairer = VizRepairer()
        self.recommender = VizRecommender()
        self.persona = PersonaExplorer()
        self.data = None
        self.infographer = None
        # Removed self.summarization_chain since Summarizer handles summarization

    def check_textgen(self, config: TextGenerationConfig):
        """
        (Optional) Method is no longer necessary as TextGenerator is removed.
        """
        pass  # Removed logic related to self.text_gen

    def summarize(self, data: Union[pd.DataFrame, str]) -> str:
        """
        Generate a summary for the provided data using Summarizer.

        Args:
            data (Union[pd.DataFrame, str]): The dataset to summarize.

        Returns:
            str: JSON-formatted summary.
        """
        return self.summarizer.summarize(data)

    def goals(
        self,
        summary: Summary,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        n: int = 5,
        persona: Persona = None
    ) -> List[Goal]:
        """
        Generate goals based on a summary.

        Args:
            summary (Summary): Summary of the dataset.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            n (int, optional): Number of goals to generate. Defaults to 5.
            persona (Persona, optional): Persona details. Defaults to None.

        Returns:
            List[Goal]: A list of generated goals.
        """
        # Removed references to self.text_gen
        if isinstance(persona, dict):
            persona = Persona(**persona)
        if isinstance(persona, str):
            persona = Persona(persona=persona, rationale="")

        # Adjust GoalExplorer.generate to no longer require text_gen
        return self.goal.generate(
            summary=summary,
            textgen_config=textgen_config,
            n=n,
            persona=persona
        )

    def personas(
        self,
        summary: Summary,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        n: int = 5
    ) -> List[Persona]:
        """
        Generate personas based on a summary.

        Args:
            summary (Summary): Summary of the dataset.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            n (int, optional): Number of personas to generate. Defaults to 5.

        Returns:
            List[Persona]: A list of generated personas.
        """
        # Removed references to self.text_gen
        return self.persona.generate(
            summary=summary,
            textgen_config=textgen_config,
            n=n
        )

    def visualize(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """
        Generate visualization code based on summary and goal.

        Args:
            summary (Summary): Summary of the dataset.
            goal (Goal): Goal for visualization.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library to use. Defaults to "seaborn".
            return_error (bool, optional): Whether to return errors. Defaults to False.

        Returns:
            Any: The generated visualization or error.
        """
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        # Removed text_gen parameter
        code_specs = self.vizgen.generate(
            summary=summary,
            goal=goal,
            library=library,
            textgen_config=textgen_config,
        )
        
        return self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )

    def execute(
        self,
        code_specs,
        data: pd.DataFrame,
        summary: Summary,
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """
        Execute the visualization code.

        Args:
            code_specs: Specifications for code generation.
            data (pd.DataFrame): The dataset.
            summary (Summary): Summary of the dataset.
            library (str, optional): Visualization library to use. Defaults to "seaborn".
            return_error (bool, optional): Whether to return errors. Defaults to False.

        Returns:
            Any: The generated visualization or error.
        """
        if data is None:
            root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
            print(root_file_path)
            data = read_dataframe(
                os.path.join(root_file_path, "files/data", summary.file_name)
            )

        return self.executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
        )

    def edit(
        self,
        code: str,
        summary: Summary,
        instructions: List[str],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """
        Edit a visualization code given a set of instructions.

        Args:
            code (str): Existing visualization code.
            summary (Summary): Summary of the dataset.
            instructions (List[str]): List of instructions for editing.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library to use. Defaults to "seaborn".
            return_error (bool, optional): Whether to return errors. Defaults to False.

        Returns:
            Any: The edited visualization or error.
        """
        # Removed check_textgen and references to self.text_gen
        if isinstance(instructions, str):
            instructions = [instructions]

        # Removed text_gen parameter
        code_specs = self.vizeditor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            textgen_config=textgen_config,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def repair(
        self,
        code: str,
        goal: Goal,
        summary: Summary,
        feedback: str,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """
        Repair a visualization given some feedback.

        Args:
            code (str): Existing visualization code.
            goal (Goal): Goal for the visualization.
            summary (Summary): Summary of the dataset.
            feedback (str): Feedback for repair.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library to use. Defaults to "seaborn".
            return_error (bool, optional): Whether to return errors. Defaults to False.

        Returns:
            Any: The repaired visualization or error.
        """
        # Removed check_textgen and references to self.text_gen
        code_specs = self.repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            textgen_config=textgen_config,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def explain(
        self,
        code: str,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """
        Explain a visualization code.

        Args:
            code (str): Visualization code to explain.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library used. Defaults to "seaborn".

        Returns:
            str: Explanation of the visualization.
        """
        # Removed check_textgen and references to self.text_gen
        return self.explainer.generate(
            code=code,
            textgen_config=textgen_config,
            library=library,
        )

    def evaluate(
        self,
        code: str,
        goal: Goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ):
        """
        Evaluate a visualization code against a goal.

        Args:
            code (str): Visualization code to evaluate.
            goal (Goal): Goal for the visualization.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library used. Defaults to "seaborn".

        Returns:
            str: Evaluation of the visualization.
        """
        # Removed check_textgen and references to self.text_gen
        return self.evaluator.generate(
            code=code,
            goal=goal,
            textgen_config=textgen_config,
            library=library,
        )

    def recommend(
        self,
        code: str,
        summary: Summary,
        n: int = 4,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """
        Recommend improvements for a visualization code.

        Args:
            code (str): Existing visualization code.
            summary (Summary): Summary of the dataset.
            n (int, optional): Number of recommendations to generate. Defaults to 4.
            textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            library (str, optional): Visualization library used. Defaults to "seaborn".
            return_error (bool, optional): Whether to return errors. Defaults to False.

        Returns:
            Any: Recommendations for improvement or error.
        """
        # Removed check_textgen and references to self.text_gen
        code_specs = self.recommender.generate(
            code=code,
            summary=summary,
            n=n,
            textgen_config=textgen_config,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def infographics(
        self, 
        visualization: str, 
        n: int = 1,
        style_prompt: Union[str, List[str]] = "",
        return_pil: bool = False
    ):
        """
        Generate infographics using the peacasso package.

        Args:
            visualization (str): Description of the visualization.
            n (int, optional): Number of infographics to generate. Defaults to 1.
            style_prompt (Union[str, List[str]], optional): Style prompts for the infographic.
            return_pil (bool, optional): Whether to return a PIL image. Defaults to False.

        Returns:
            Any: Generated infographic or error.
        """
        try:
            import peacasso
        except ImportError as exc:
            raise ImportError(
                'Please install lida with infographics support. pip install lida[infographics]. You will also need a GPU runtime.'
            ) from exc

        from ..components.infographer import Infographer

        if self.infographer is None:
            logger.info("Initializing Infographer")
            self.infographer = Infographer()
        return self.infographer.generate(
            visualization=visualization, 
            n=n, 
            style_prompt=style_prompt, 
            return_pil=return_pil
        )
