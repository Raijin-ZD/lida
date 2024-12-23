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
            
        # Rule 4: Check for Dask DataFrame handling if not datashader
        if "datashader" not in code:
            dask_check = "if isinstance(data, dd.DataFrame):\n        data = data.sample(frac=0.1, random_state=42).compute()"
            if dask_check not in code:
                issues.append("Missing Dask DataFrame handling")
        
        # Rule 5: Check syntax
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
            if plot_idx >= 0:
                code_lines.insert(plot_idx + 1, 
                    "    if isinstance(data, dd.DataFrame):\n        data = data.sample(frac=0.1, random_state=42).compute()")
                fixed_issues.append("Added Dask DataFrame handling")
        
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
    """Hybrid code repair agent using rules first, then LangChain."""
    
    def __init__(self, text_gen: TextGenerator, textgen_config: TextGenerationConfig):
        self.rule_based = RuleBasedRepair()
        
        prompt_template = """Fix this Python visualization code. Keep all imports and maintain code structure.
        The fixed code must have:
        1. All required imports (pandas, matplotlib, dask)
        2. A plot(data) function
        3. Dask DataFrame handling
        4. chart = plot(data) assignment
        
        Code to fix:
        {code}
        
        Return only the fixed code without any comments or explanations."""
        
        self.chain = LLMChain(
            llm=TextGeneratorLLM(
                text_gen=text_gen,
                system_prompt="You are a Python code repair expert. Fix code while preserving structure and imports.",
                temperature=textgen_config.temperature,
                max_tokens=textgen_config.max_tokens,
            ),
            prompt=PromptTemplate(
                input_variables=["code"],
                template=prompt_template
            )
        )

    def _clean_langchain_output(self, output: str) -> str:
        """Clean and extract code from LangChain output."""
        if not output:
            return ""
            
        # Remove markdown code blocks if present
        if "```" in output:
            pattern = r"```(?:python)?\s*(.*?)\s*```"
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                output = matches[0]
        
        # Remove any comments
        output = re.sub(r'#.*$', '', output, flags=re.MULTILINE)
        
        # Remove extra whitespace and blank lines
        output = '\n'.join(line for line in output.split('\n') if line.strip())
        
        # Extract just the code if there's any text before/after
        if 'def plot(data):' in output:
            code_start = output.find('import ') if 'import ' in output else output.find('def plot')
            code_end = output.rfind('chart = plot(data)') + len('chart = plot(data)')
            output = output[code_start:code_end]
            
        return output.strip()

    def repair(self, faulty_code: str) -> str:
        """Repair code using rules first, then LangChain if needed."""
        logger.info("Starting repair process...")
        
        # Try rule-based fixes first
        fixed_code, success, issues = self.rule_based.repair_code(faulty_code)
        
        if success:
            logger.info("Code fixed with rules or already valid")
            return fixed_code
            
        # Use LangChain for complex fixes
        logger.info("Rule-based repair failed, attempting LangChain fix")
        try:
            # Get imports from original code
            original_imports = self.rule_based._extract_imports(faulty_code)
            
            # Run LangChain
            chain_response = self.chain.run(code=faulty_code)
            
            # Clean response and ensure imports
            cleaned_code = chain_response.strip()
            if '```' in cleaned_code:
                cleaned_code = re.findall(r'```(?:python)?\n(.*?)\n```', cleaned_code, re.DOTALL)[0]
            
            # Preserve original imports
            existing_imports = self.rule_based._extract_imports(cleaned_code)
            for imp in original_imports:
                if imp not in existing_imports:
                    cleaned_code = imp + '\n' + cleaned_code
            
            # Validate and potentially fix with rules
            is_valid, validation_issues = self.rule_based.validate_code(cleaned_code)
            if is_valid:
                return cleaned_code
                
            # Try one more time with rule-based repair
            final_code, success, _ = self.rule_based.repair_code(cleaned_code)
            if success:
                return final_code
                
            logger.error(f"LangChain fix failed validation: {validation_issues}")
            return fixed_code
            
        except Exception as e:
            logger.error(f"LangChain repair failed: {e}")
            return fixed_code
