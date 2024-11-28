# textgen_langchain.py

from langchain.llms.base import LLM
from typing import Optional, List, Any
from llmx import TextGenerator
from lida.datamodel import TextGenerationConfig
import logging
from pydantic import ConfigDict, model_validator

logger = logging.getLogger("lida")

from pydantic import PrivateAttr

class TextGeneratorLLM(LLM):
    _text_gen: TextGenerator = PrivateAttr()
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    system_prompt: str = ""  # Add this line
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, text_gen: TextGenerator,system_prompt, **kwargs):
        super().__init__(**kwargs)
        self._text_gen = text_gen
        self.system_prompt = system_prompt  # Store the system prompt


    def _llm_type(self) -> str:
        return "text_generator_llm"
    

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            config = TextGenerationConfig(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop or self.stop,
            )
            messages = [{'role': 'user', 'content': prompt},
                        {'role': 'system', 'content': self.llm.system_prompt},
]
            print(f"Messages sent to text_gen.generate: {messages}")
            print(f"Config: {config}")
            
            response = self._text_gen.generate(
                messages==messages,
                config=config,
            )

            print(f"Response from text_gen.generate: {response}")

            # Extract generated text based on response structure
            if hasattr(response, 'text'):
                if isinstance(response.text, list):
                    # Extract content from the first message if needed
                    if len(response.text) > 0 and hasattr(response.text[0], 'content'):
                        generated_text = response.text[0].content
                    else:
                        generated_text = ''
                elif isinstance(response.text, str):
                    generated_text = response.text
                else:
                    generated_text = ''
            else:
                generated_text = ''

            # Remove stop sequences if provided
            if stop is not None:
                for stop_token in stop:
                    generated_text = generated_text.split(stop_token)[0]

            return generated_text.strip()
        except Exception as e:
            print(f"Error in TextGeneratorLLM _call: {e}")
            raise e

    @property
    def _identifying_params(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def set_params(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, param, value)
