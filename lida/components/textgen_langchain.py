# textgen_langchain.py

from langchain.llms.base import LLM
from typing import Optional, List
from llmx import TextGenerator
from lida.datamodel import TextGenerationConfig
import logging

logger = logging.getLogger("lida")

class TextGeneratorLLM(LLM):
    text_gen: TextGenerator  # Declare as a class attribute
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

    def __init__(self, text_gen: TextGenerator, **kwargs):
        """
        Wrapper class to make TextGenerator compatible with LangChain's LLM interface.

        Args:
            text_gen (TextGenerator): An instance of TextGenerator from llmx.
        """
        super().__init__(**kwargs)  # Call the parent constructor
        self.text_gen = text_gen    # Assign the text generator

    @property
    def _llm_type(self) -> str:
        return "llmx_text_generator"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Makes a call to the text generation API using the prompt.

        Args:
            prompt (str): The prompt string.
            stop (List[str], optional): A list of stop tokens.

        Returns:
            str: The generated text.
        """
        try:
            # Create a TextGenerationConfig object with the necessary parameters
            config = TextGenerationConfig(
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                stop=stop or self.stop,
            )

            # Pass the prompt as a string
            messages = prompt

            # If TextGenerator expects messages as a list of dicts, uncomment the following:
            # messages = [{"role": "user", "content": prompt}]

            # Call the generate method of TextGenerator
            response = self.text_gen.generate(
                messages=messages,
                config=config,
            )

            # Extract the generated text from the response
            if hasattr(response, 'text'):
                # Assuming response.text is a string; adjust if it's a different structure
                generated_text = response.text.strip()
                return generated_text
            else:
                logger.error("Response does not contain 'text' attribute.")
                return ""
        except Exception as e:
            logger.error(f"Error in TextGeneratorLLM _call: {e}")
            return ""

    @property
    def _identifying_params(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def set_params(self, **kwargs):
        """
        Set parameters for the LLM.
        """
        for param, value in kwargs.items():
            setattr(self, param, value)
