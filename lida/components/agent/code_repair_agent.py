from ..textgen_langchain import TextGeneratorLLM
from llmx import TextGenerator, TextGenerationConfig
from langchain.agents import Tool, initialize_agent, AgentType
import ast
import logging

# Add detailed logging configuration
logger = logging.getLogger("lida.code_repair")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to see all logs

class CodeRepairAgent:
    """An agent that repairs code using LangChain"""

    def __init__(self, text_gen: TextGenerator, textgen_config: TextGenerationConfig):
        # Initialize LLM with proper configuration
        self.text_gen = text_gen
        self.textgen_config = textgen_config
        self.llm = TextGeneratorLLM(
            text_gen=text_gen,
            system_prompt="""You are an expert Python developer tasked with repairing visualization code.
            Follow these rules:
            1. Ensure code has a plot(data) function
            2. The function must return a visualization object
            3. Only use standard visualization libraries
            4. Handle both Pandas and Dask DataFrames
            5. Return only the fixed code, no explanations""",
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
            ),
            Tool(
                name="validate_chart_creation",
                func=self._validate_chart_creation,
                description="Validate that code creates and returns a chart object"
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
        self.debug = True  # Add debug flag
        self.timeout_seconds = 60  # Add timeout limit
        logger.info("CodeRepairAgent initialized with debugging enabled")

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
                    return "Plot function structure is valid"
            return "No plot function found"
        except Exception as e:
            return f"Validation error: {str(e)}"

    def _validate_chart_creation(self, code: str) -> str:
        """Validate that code creates and returns a chart object"""
        try:
            tree = ast.parse(code)
            issues = []
            has_chart_assignment = False
            has_chart_return = False
            
            for node in ast.walk(tree):
                # Check for chart assignment
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'chart':
                            has_chart_assignment = True
                
                # Check for chart return in plot function
                if isinstance(node, ast.FunctionDef) and node.name == 'plot':
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Return):
                            if hasattr(sub_node.value, 'id'):  # Direct variable return
                                if sub_node.value.id in ['chart', 'fig', 'plt']:
                                    has_chart_return = True
                            elif isinstance(sub_node.value, ast.Name):  # Return statement exists
                                has_chart_return = True

            if not has_chart_assignment:
                issues.append("No 'chart' object is created")
            if not has_chart_return:
                issues.append("Plot function doesn't return a chart object")
                
            return "Chart creation validation issues:\n" + "\n".join(issues) if issues else "Chart creation is valid"
            
        except Exception as e:
            return f"Chart validation error: {str(e)}"

    def _validate_plot_performance(self, code: str) -> str:
        """Validate plot performance and data handling"""
        issues = []
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            # Check for nested scatterplots with same hue
            if isinstance(node, ast.Call) and hasattr(node.func, 'value'):
                if node.func.value.id == 'sns' and node.func.attr == 'scatterplot':
                    for kw in node.keywords:
                        if kw.arg == 'hue':
                            issues.append("Multiple scatterplots with hue may cause conflicts")
                            
            # Check for small data sampling
            if isinstance(node, ast.Call) and hasattr(node.func, 'value'):
                if node.func.value.id == 'data' and node.func.attr == 'sample':
                    for kw in node.keywords:
                        if kw.arg == 'frac' and kw.value.value > 0.1:
                            issues.append("Data sampling fraction too large, should be <= 0.1")
                            
        return "\n".join(issues) if issues else "Performance checks passed"

    def _repair_code(self, code: str, issue_type: str, error_msg: str) -> str:
        """Helper method to repair code based on issue type"""
        print(f"🔧 Attempting {issue_type} repair...")
        
        # Add performance fixes to the system prompt
        performance_fixes = """
        - Use smaller data samples (frac=0.1) for large datasets
        - Avoid multiple scatterplots with hue parameter
        - Add plt.figure(figsize=(10,6)) before plotting
        - Use alpha=0.5 for transparency
        - Clear the plot after each attempt with plt.clf()
        """
        
        messages = [
            {"role": "system", "content": f"""You are an expert Python visualization code repairer.
            Fix the code following these performance guidelines:
            {performance_fixes}
            Return ONLY the fixed code without explanations."""},
            {"role": "user", "content": f"""
            Fix this {issue_type} issue: {error_msg}
            
            Current code:
            {code}
            
            If using seaborn scatterplots with hue, combine them into a single plot.
            Use smaller data samples for better performance.
            Return only the fixed code."""}
        ]
        
        try:
            response = self.text_gen.generate(messages=messages, config=self.textgen_config)
            if response and hasattr(response, 'text') and len(response.text) > 0:
                fixed_code = response.text[0]['content']
                # Add performance checks
                if 'sns.scatterplot' in fixed_code:
                    fixed_code = fixed_code.replace(
                        'def plot(data):',
                        'def plot(data):\n    plt.figure(figsize=(10,6))\n    if isinstance(data, dd.DataFrame):\n        data = data.sample(frac=0.1, random_state=42).compute()\n    else:\n        data = data.sample(frac=0.1, random_state=42)'
                    )
                print(f"After {issue_type} repair:\n{fixed_code}")
                return fixed_code
        
        except Exception as e:
            print(f"Error in repair: {str(e)}")
            return code
            
        return code

    def repair(self, code: str) -> str:
        """Repair code through multiple attempts if needed"""
        current_code = code
        print("\n=== Starting Code Repair Process ===")
        print(f"Original code to repair:\n{code}")
        
        for attempt in range(self.max_repair_attempts):
            try:
                print(f"\n--- Repair Attempt {attempt + 1} ---")
                
                # Validate code in sequence
                validations = [
                    (self._validate_syntax, "syntax"),
                    (self._validate_plot_function, "plot_function"),
                    (self._validate_chart_creation, "chart_creation")
                ]
                
                # Add performance validation
                validations.append((self._validate_plot_performance, "performance"))
                
                for validate_func, validation_type in validations:
                    result = validate_func(current_code)
                    print(f"{validation_type.title()} check result: {result}")
                    
                    if ("error" in result.lower() or 
                        "valid" not in result.lower() or 
                        "issues" in result.lower()):
                        current_code = self._repair_code(current_code, validation_type, result)
                        # Skip remaining validations for this attempt
                        break
                
                # If we complete all validations, we're done
                if all(validate_func(current_code)[0] not in ["error", "issues"] 
                      for validate_func, _ in validations):
                    if current_code != code:
                        print("\n✅ Code was successfully repaired!")
                        print(f"Final code:\n{current_code}")
                    else:
                        print("\n✅ Code passed all validations")
                    return current_code

            except Exception as e:
                print(f"❌ Error in repair attempt {attempt + 1}: {str(e)}")
                continue

        print("\n⚠️ Max repair attempts reached")
        return current_code