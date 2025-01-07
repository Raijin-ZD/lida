from ..textgen_langchain import TextGeneratorLLM
from llmx import TextGenerator, TextGenerationConfig
from langchain.agents import Tool, AgentType, initialize_agent, ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import ast
import logging
import re

logger = logging.getLogger("code_repair_agent")
logger.setLevel(logging.DEBUG)
print("Code repair agent loaded")
class RuleBasedRepair:
    """Handles basic code structure and syntax validation/repair."""
    
    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements from code."""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                    lines = code.split('\n')[start:end]
                    imports.extend(lines)
        except:
            # Fallback to regex if AST parse fails
            import_pattern = r'^(?:from [^import]+ )?import [^;]+$'
            imports = [line.strip() for line in code.split('\n') 
                      if re.match(import_pattern, line.strip())]
        return imports

    def validate_code(self, code: str) -> tuple[bool, list[str]]:
        """Validate code against all rules."""
        issues = []
        
        # Rule 1: Check for required imports
        required_imports = {
            'pandas': 'import pandas as pd',
            'matplotlib': 'import matplotlib.pyplot as plt',
            'dask': 'import dask.dataframe as dd'
        }
        
        existing_imports = self._extract_imports(code)
        missing_imports = []
        for imp in required_imports.values():
            if not any(imp in exist_imp for exist_imp in existing_imports):
                missing_imports.append(imp)
                issues.append(f"Missing import: {imp}")
        
        # Rule 2: Check for plot function
        if "def plot(data):" not in code:
            issues.append("Missing plot(data) function")
            
        # Rule 3: Check for chart = plot(data)
        if "chart = plot(data)" not in code:
            issues.append("Missing chart = plot(data) assignment")
            
        
        # Rule 4: Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            
        return len(issues) == 0, issues

    def repair_code(self, code: str) -> tuple[str, bool, list[str]]:
        """Apply basic repairs to code."""
        is_valid, issues = self.validate_code(code)
        if is_valid:
            return code, True, []
            
        fixed_code = code
        fixed_issues = []
        
        # First preserve existing imports
        import_lines = self._extract_imports(fixed_code)
        required_imports = [
            'import pandas as pd',
            'import matplotlib.pyplot as plt',
            'import dask.dataframe as dd'
        ]
        
        # Add any missing required imports
        for imp in required_imports:
            if not any(imp in existing for existing in import_lines):
                import_lines.append(imp)
                fixed_issues.append(f"Added missing import: {imp}")
        
        # Get non-import code
        code_lines = [line for line in fixed_code.split('\n') 
                     if line.strip() and not any(line.strip() == imp for imp in import_lines)]
        
        if not any(line.strip().startswith('def plot(data):') for line in code_lines):
            code_lines.insert(0, "\ndef plot(data):\n    return plt")
            fixed_issues.append("Added plot function")
        
        # Add Dask handling if needed
        if "datashader" not in '\n'.join(code_lines):
            plot_idx = next((i for i, line in enumerate(code_lines) 
                           if line.strip().startswith('def plot(data):')), -1)
        
        # Ensure chart assignment exists
        if not any('chart = plot(data)' in line for line in code_lines):
            code_lines.append('\nchart = plot(data)')
            fixed_issues.append("Added chart assignment")
        
        # Reconstruct code with proper spacing
        fixed_code = '\n'.join(import_lines) + '\n\n' + '\n'.join(code_lines)
        
        try:
            ast.parse(fixed_code)  # Validate syntax
            return fixed_code, True, fixed_issues
        except SyntaxError:
            return code, False, ["Syntax error could not be fixed with rules"]

class CodeRepairAgent:
    def __init__(self, text_gen: TextGenerator, textgen_config: TextGenerationConfig):
        self.rule_based = RuleBasedRepair()
        
        # Define tools for code repair
        self.tools = [
            Tool(
                name="fix_syntax",
                func=self._fix_syntax_errors,
                description="Fix Python syntax errors in code"
            ),
            Tool(
                name="validate_structure",
                func=self._validate_code_structure,
                description="Check if code has required plot function and chart assignment"
            ),
            Tool(
                name="clean_code",
                func=self._clean_code,
                description="Remove unnecessary comments and clean code formatting"
            )
        ]

        # Create agent prompt
        prompt = ZeroShotAgent.create_prompt(
            tools=self.tools,
            prefix="""You are a Python code repair expert. Fix visualization code while preserving structure.
            The code must have:
            1. Valid Python syntax
            2. plot(data) function
            3. chart = plot(data) assignment
            
            Analyze the code and use available tools to fix issues.""",
            suffix="Code: {input}\nThought: Let's approach this step by step.",
            input_variables=["input"]
        )

        # Initialize LLM
        llm = TextGeneratorLLM(
            text_gen=text_gen,
            system_prompt="You are a code repair expert.",
            temperature=textgen_config.temperature,
            max_tokens=textgen_config.max_tokens
        )

        # Create agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def _fix_syntax_errors(self, code: str) -> str:
        """Tool to fix syntax errors"""
        try:
            ast.parse(code)
            return "Code syntax is valid"
        except SyntaxError as e:
            return f"Found syntax error: {str(e)}"

    def _validate_code_structure(self, code: str) -> str:
        """Tool to validate code structure"""
        issues = []
        if "def plot(data):" not in code:
            issues.append("Missing plot(data) function")
        if "chart = plot(data)" not in code:
            issues.append("Missing chart assignment")
        return str(issues) if issues else "Code structure is valid"

    def _clean_code(self, code: str) -> str:
        """Tool to clean code"""
        return self._clean_langchain_output(code)

    def repair(self, faulty_code: str) -> str:
        """Repair code using rule-based first, then agent if needed"""
        logger.info("Starting repair process...")
        
        # Try rule-based first
        fixed_code, success, issues = self.rule_based.repair_code(faulty_code)
        
        if success:
            logger.info("Code fixed with rules or already valid")
            return fixed_code
            
        # Use agent for complex fixes
        logger.info("Rule-based repair failed, attempting Agent fix")
        try:
            agent_response = self.agent.run({
                "input": faulty_code,
                "issues": issues
            })
            
            # Clean and validate agent output
            cleaned_code = self._clean_langchain_output(agent_response)
            is_valid, validation_issues = self.rule_based.validate_code(cleaned_code)
            
            if is_valid:
                return cleaned_code
                
            logger.error(f"Agent fix failed validation: {validation_issues}")
            return fixed_code
            
        except Exception as e:
            logger.error(f"Agent repair failed: {e}")
            return fixed_code
