"""Result classes and formatting for CLI output."""

import json
from typing import Dict, Any


class VerificationResult:
    """Result of a verification attempt."""
    
    def __init__(self, function_name: str, success: bool, 
                 message: str = "", statistics: Dict[str, Any] = None,
                 counterexample: Dict[str, Any] = None):
        self.function_name = function_name
        self.success = success
        self.message = message
        self.statistics = statistics or {}
        self.counterexample = counterexample
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function_name,
            "success": self.success,
            "message": self.message,
            "statistics": self.statistics,
            "counterexample": self.counterexample
        }


def format_verification_result(result: VerificationResult, output_format: str = "text") -> str:
    """Format a verification result for output."""
    if output_format == "json":
        return json.dumps(result.to_dict(), indent=2)
    elif output_format == "text":
        if result.success:
            icon = "✓"
            color = "green"
        else:
            icon = "✗"
            color = "red"
        
        lines = [f"{icon} {result.function_name}: {result.message}"]
        
        if result.counterexample:
            lines.append("  Counterexample:")
            for var, value in result.counterexample.items():
                lines.append(f"    {var} = {value}")
        
        if result.statistics:
            lines.append("  Statistics:")
            for stat, value in result.statistics.items():
                lines.append(f"    {stat}: {value}")
        
        return "\n".join(lines)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
