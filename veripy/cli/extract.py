"""Extract verification information from Python files."""

import ast
from pathlib import Path
from typing import Dict, Any


def extract_verification_info(file_path: Path) -> Dict[str, Any]:
    """Extract verification-related information from a Python file."""
    info = {
        "functions": [],
        "requires": [],
        "ensures": [],
        "invariants": []
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "requires": [],
                    "ensures": [],
                    "decreases": None
                }
                
                # Look for decorators that might be verification-related
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if decorator.func.id == "verify":
                                # Extract arguments from verify decorator
                                for keyword in decorator.keywords:
                                    if keyword.arg == "requires":
                                        if isinstance(keyword.value, ast.List):
                                            for elt in keyword.value.elts:
                                                if isinstance(elt, ast.Constant):
                                                    func_info["requires"].append(elt.value)
                                    elif keyword.arg == "ensures":
                                        if isinstance(keyword.value, ast.List):
                                            for elt in keyword.value.elts:
                                                if isinstance(elt, ast.Constant):
                                                    func_info["ensures"].append(elt.value)
                                    elif keyword.arg == "decreases":
                                        if isinstance(keyword.value, ast.Constant):
                                            func_info["decreases"] = keyword.value.value
                
                info["functions"].append(func_info)
            
            # Look for invariant calls in the code
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id == "invariant":
                            if node.args and isinstance(node.args[0], ast.Constant):
                                info["invariants"].append({
                                    "line": node.lineno,
                                    "expression": node.args[0].value
                                })
    
    except Exception as e:
        info["error"] = str(e)
    
    return info
