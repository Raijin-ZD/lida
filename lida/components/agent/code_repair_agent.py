from ..textgen_langchain import TextGeneratorLLM
from llmx import TextGenerator, TextGenerationConfig
from langchain.agents import Tool, initialize_agent, AgentType
import ast
import logging

logger = logging.getLogger("lida")

class CodeRepairAgent:
    """An agent that repairs code using LangChain"""

    def __init__(self, text_gen: TextGenerator, textgen_config: TextGenerationConfig):
        self.llm = TextGeneratorLLM(
            text_gen=text_gen,
            system_prompt="""You are an expert Python developer tasked with repairing visualization code.
            Follow these rules:
            1. Ensure code has a plot(data) function
            2. The function must return a visualization object
            3. Only use standard visualization libraries
            4. Handle both Pandas and Dask DataFrames
            5. For datashader:
               - Use tf.shade() for single plots
               - Use tf.stack() for combining multiple plots
               - Return the final image directly
            6. Return only the fixed code, no explanations""",
            temperature=textgen_config.temperature,
            max_tokens=textgen_config.max_tokens
        )
        
        # Add code validation tools
        self.tools = [
            Tool(
                name="validate_syntax",
                func=self._validate_syntax,
                description="Check Python code syntax and return errors if any"
            ),
            Tool(
                name="validate_plot_function",
                func=self._validate_plot_function,
                description="Validate plot function structure and requirements"
            )
        ]

        # Initialize agent
        self.agent_chain = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        self.max_repair_attempts = 3

        # Add debug mode
        self.debug = True
        logger.setLevel(logging.DEBUG)

    def _print_diff(self, old_code: str, new_code: str):
        """Print differences between old and new code"""
        import difflib
        logger.info("Code changes:")
        for line in difflib.unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile='before',
            tofile='after'
        ):
            logger.info(line.rstrip())

    def _validate_syntax(self, code: str) -> str:
        """Validate code syntax"""
        try:
            ast.parse(code)
            return "Code syntax is valid"
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"

    def _validate_plot_function(self, code: str) -> str:
        """Validate plot function structure"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "plot":
                    # Check if function has data parameter
                    if not node.args.args or node.args.args[0].arg != "data":
                        return "Plot function must have 'data' parameter"
                    # Check if function has return statement
                    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                    if not has_return:
                        return "Plot function must have return statement"
                    
                    # Add datashader-specific validation
                    if "datashader" in code:
                        if "tf.Images" in code:
                            return "Invalid datashader usage: Use tf.stack() instead of tf.Images for combining plots"
                    
                    return "Plot function structure is valid"
            return "No plot function found"
        except Exception as e:
            return f"Validation error: {str(e)}"

    def repair(self, code: str) -> str:
        """Repair code through multiple attempts if needed"""
        current_code = code
        
        logger.info("=" * 50)
        logger.info("STARTING CODE REPAIR PROCESS")
        logger.info("=" * 50)
        logger.info(f"Original code:\n{code}")
        
        for attempt in range(self.max_repair_attempts):
            try:
                logger.info(f"\nATTEMPT {attempt + 1}/{self.max_repair_attempts}")
                logger.info("-" * 30)

                # Check syntax
                syntax_result = self._validate_syntax(current_code)
                logger.info(f"Syntax check result: {syntax_result}")
                
                if "error" in syntax_result.lower():
                    logger.info("🔧 Attempting syntax fix...")
                    old_code = current_code
                    current_code = self.agent_chain.run({
                        "input": f"Fix this code syntax error: {syntax_result}\n\nCode:\n{current_code}"
                    })
                    self._print_diff(old_code, current_code)
                    continue

                # Check plot function
                plot_result = self._validate_plot_function(current_code)
                logger.info(f"Plot function check result: {plot_result}")
                
                if "valid" not in plot_result.lower():
                    logger.info("🔧 Attempting plot function fix...")
                    old_code = current_code
                    current_code = self.agent_chain.run({
                        "input": f"Fix this plot function issue: {plot_result}\n\nCode:\n{current_code}"
                    })
                    self._print_diff(old_code, current_code)
                    continue

                logger.info("✅ Code repair successful!")
                return current_code

            except Exception as e:
                logger.error(f"❌ Error in repair attempt {attempt + 1}: {str(e)}")
                continue

        logger.warning("⚠️ Max repair attempts reached without full validation")
        return current_code