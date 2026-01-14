"""
Comprehensive Error Reporting and Counterexample Generation

This module provides detailed error reporting with source locations,
counterexamples, and suggestions for fixing verification failures.
"""

import ast
import z3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import inspect
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.markup import escape


class ErrorSeverity(Enum):
    """Severity levels for verification errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of verification errors."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    LOOP_INVARIANT = "loop_invariant"
    TERMINATION = "termination"
    TYPE_ERROR = "type_error"
    FRAME_ERROR = "frame_error"
    ASSERTION = "assertion"
    UNKNOWN = "unknown"


@dataclass
class SourceLocation:
    """Source code location information."""
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.column}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column
        }


@dataclass
class Counterexample:
    """Counterexample showing a violation."""
    values: Dict[str, Any]
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "explanation": self.explanation
        }
    
    def format(self) -> str:
        lines = ["Counterexample:"]
        for var, value in self.values.items():
            lines.append(f"  {var} = {value}")
        if self.explanation:
            lines.append(f"\nExplanation: {self.explanation}")
        return "\n".join(lines)


@dataclass
class VerificationError:
    """A verification error with full context."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    location: Optional[SourceLocation] = None
    expression: Optional[str] = None
    counterexample: Optional[Counterexample] = None
    suggestion: Optional[str] = None
    related_errors: List["VerificationError"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "expression": self.expression,
            "suggestion": self.suggestion,
            "related_errors": [e.to_dict() for e in self.related_errors]
        }
        if self.location:
            result["location"] = self.location.to_dict()
        if self.counterexample:
            result["counterexample"] = self.counterexample.to_dict()
        return result


class ErrorReporter:
    """
    Comprehensive error reporter for verification failures.
    
    Provides:
    - Rich terminal output with syntax highlighting
    - Detailed error messages with source locations
    - Counterexample generation and formatting
    - Suggestions for fixing errors
    - Multiple output formats (text, JSON)
    """
    
    def __init__(self, show_suggestions: bool = True, 
                 show_counterexamples: bool = True):
        self.console = Console()
        self.show_suggestions = show_suggestions
        self.show_counterexamples = show_counterexamples
        self.errors: List[VerificationError] = []
        self.warnings: List[VerificationError] = []
        self.statistics = {
            "total_errors": 0,
            "total_warnings": 0,
            "errors_by_category": {cat: 0 for cat in ErrorCategory},
            "verification_time": 0.0
        }
    
    def add_error(self, error: VerificationError):
        """Add a verification error to the report."""
        self.errors.append(error)
        self.statistics["total_errors"] += 1
        self.statistics["errors_by_category"][error.category] += 1
    
    def add_warning(self, warning: VerificationError):
        """Add a verification warning to the report."""
        self.warnings.append(warning)
        self.statistics["total_warnings"] += 1
    
    def extract_location(self, frame_info: inspect.FrameInfo) -> SourceLocation:
        """Extract source location from a frame info."""
        return SourceLocation(
            file=frame_info.filename,
            line=frame_info.lineno,
            column=frame_info.column
        )
    
    def generate_counterexample(self, model: z3.Model, 
                               variables: Dict[str, Any]) -> Counterexample:
        """Generate a counterexample from an SMT model."""
        values = {}
        
        for var_name, var_type in variables.items():
            try:
                if var_name in model:
                    decl = model[var_name]
                    if decl is not None:
                        values[var_name] = str(decl)
                    else:
                        values[var_name] = "unconstrained"
                else:
                    values[var_name] = "not in model"
            except Exception:
                values[var_name] = "error reading value"
        
        explanation = self._explain_counterexample(values)
        
        return Counterexample(values=values, explanation=explanation)
    
    def _explain_counterexample(self, values: Dict[str, Any]) -> str:
        """Generate an explanation for a counterexample."""
        explanations = []
        
        for var, value in values.items():
            if value in ("unconstrained", "not in model"):
                explanations.append(f"{var} has no constraints")
            else:
                try:
                    num_val = int(value)
                    if abs(num_val) > 1000:
                        explanations.append(f"{var} has a large value ({num_val})")
                except ValueError:
                    pass
        
        if explanations:
            return "; ".join(explanations)
        return "No specific explanation available"
    
    def format_error_text(self, error: VerificationError) -> str:
        """Format a single error for text output."""
        lines = []
        
        # Category and severity
        category = error.category.value.upper()
        severity_icon = {
            ErrorSeverity.INFO: "ℹ",
            ErrorSeverity.WARNING: "⚠",
            ErrorSeverity.ERROR: "✗",
            ErrorSeverity.CRITICAL: "⛔"
        }.get(error.severity, "✗")
        
        lines.append(f"{severity_icon} [{category}] {error.message}")
        
        # Location
        if error.location:
            lines.append(f"  Location: {error.location}")
        
        # Expression
        if error.expression:
            lines.append(f"  Expression: {error.expression}")
        
        # Counterexample
        if error.counterexample and self.show_counterexamples:
            lines.append(f"\n{error.counterexample.format()}")
        
        # Suggestion
        if error.suggestion and self.show_suggestions:
            lines.append(f"\n  💡 Suggestion: {error.suggestion}")
        
        # Related errors
        if error.related_errors:
            lines.append("\n  Related errors:")
            for related in error.related_errors:
                lines.append(f"    - {related.message}")
        
        return "\n".join(lines)
    
    def format_source_context(self, file_path: str, line: int, 
                             column: int, context_lines: int = 3) -> str:
        """Format source code context around an error location."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if line < 1 or line > len(lines):
                return f"Unable to read source context (line {line} out of range)"
            
            start = max(1, line - context_lines)
            end = min(len(lines), line + context_lines)
            
            result = []
            for i in range(start, end + 1):
                prefix = ">>>" if i == line else "   "
                line_content = lines[i - 1].rstrip()
                result.append(f"{prefix} {i:4d} │ {line_content}")
            
            return "\n".join(result)
        
        except Exception as e:
            return f"Unable to read source context: {e}"
    
    def format_report(self, output_format: str = "text") -> str:
        """Format the full verification report."""
        if output_format == "json":
            return self._format_json_report()
        else:
            return self._format_text_report()
    
    def _format_text_report(self) -> str:
        """Format report as human-readable text."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("VERIFICATION REPORT")
        lines.append("=" * 60)
        
        # Summary
        lines.append(f"\nSummary:")
        lines.append(f"  Total errors: {self.statistics['total_errors']}")
        lines.append(f"  Total warnings: {self.statistics['total_warnings']}")
        lines.append(f"  Verification time: {self.statistics['verification_time']:.2f}s")
        
        # Errors by category
        if self.statistics['total_errors'] > 0:
            lines.append(f"\nErrors by category:")
            for cat, count in self.statistics['errors_by_category'].items():
                if count > 0:
                    lines.append(f"  {cat.value}: {count}")
        
        # Detailed errors
        if self.errors:
            lines.append(f"\n{'=' * 60}")
            lines.append("DETAILED ERRORS")
            lines.append('=' * 60)
            
            for i, error in enumerate(self.errors, 1):
                lines.append(f"\n[{i}] {self.format_error_text(error)}")
                
                # Source context
                if error.location:
                    source = self.format_source_context(
                        error.location.file,
                        error.location.line,
                        error.location.column
                    )
                    if source:
                        lines.append(f"\n  Source context:")
                        lines.append(source)
        
        # Warnings
        if self.warnings:
            lines.append(f"\n{'=' * 60}")
            lines.append("WARNINGS")
            lines.append('=' * 60)
            
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"\n[{i}] {self.format_error_text(warning)}")
        
        # Footer
        lines.append(f"\n{'=' * 60}")
        if self.errors:
            lines.append("VERIFICATION FAILED")
        else:
            lines.append("VERIFICATION PASSED")
        lines.append('=' * 60)
        
        return "\n".join(lines)
    
    def _format_json_report(self) -> str:
        """Format report as JSON."""
        import json
        
        # Convert enums to serializable primitives
        errors_by_category = {
            cat.value if isinstance(cat, ErrorCategory) else str(cat): count
            for cat, count in self.statistics["errors_by_category"].items()
        }
        summary = dict(self.statistics)
        summary["errors_by_category"] = errors_by_category

        report = {
            "summary": summary,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "passed": len(self.errors) == 0
        }
        
        return json.dumps(report, indent=2)
    
    def print_report(self, output_format: str = "text"):
        """Print the verification report to the console."""
        if output_format == "text":
            self.console.print(self._format_text_report())
        elif output_format == "json":
            self.console.print(Syntax(self._format_json_report(), "json"))
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def create_error_from_exception(self, exc: Exception, 
                                   category: ErrorCategory = ErrorCategory.UNKNOWN,
                                   location: Optional[SourceLocation] = None) -> VerificationError:
        """Create a verification error from an exception."""
        return VerificationError(
            category=category,
            severity=ErrorSeverity.ERROR,
            message=str(exc),
            location=location,
            suggestion=self._suggest_fix(exc, category)
        )
    
    def _suggest_fix(self, exc: Exception, category: ErrorCategory) -> str:
        """Generate a suggestion for fixing an error."""
        suggestions = {
            ErrorCategory.PRECONDITION: (
                "Precondition not satisfied. Consider adding a precondition check "
                "or adjusting the function's @verify(requires=[...]) specification."
            ),
            ErrorCategory.POSTCONDITION: (
                "Postcondition not satisfied. Consider adding intermediate assertions "
                "or adjusting the function's @verify(ensures=[...]) specification."
            ),
            ErrorCategory.LOOP_INVARIANT: (
                "Loop invariant not maintained. Consider adding more specific invariants "
                "using the invariant(...) function within the loop body."
            ),
            ErrorCategory.TERMINATION: (
                "Recursive function may not terminate. Consider adding a decreases clause "
                "to @verify(decreases=...) or restructuring the recursion."
            ),
            ErrorCategory.TYPE_ERROR: (
                "Type error detected. Check that expressions use compatible types "
                "and consider adding type annotations."
            ),
            ErrorCategory.FRAME_ERROR: (
                "Frame condition violated. Ensure that modified variables are declared "
                "in the @verify(modifies=[...]) specification."
            ),
            ErrorCategory.ASSERTION: (
                "Assertion failed. Check the assertion expression or consider "
                "adding a lemma using assert_by(...) for complex properties."
            ),
            ErrorCategory.UNKNOWN: (
                "An unknown error occurred. Check the error message for details."
            )
        }
        
        return suggestions.get(category, suggestions[ErrorCategory.UNKNOWN])


def extract_source_location(node: ast.AST) -> SourceLocation:
    """Extract source location from an AST node."""
    if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
        return SourceLocation(
            file=getattr(node, 'filename', '<unknown>'),
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, 'end_lineno', None),
            end_column=getattr(node, 'end_col_offset', None)
        )
    return None


def parse_smt_model(model: z3.Model) -> Dict[str, Any]:
    """Parse an SMT model into a dictionary."""
    result = {}
    
    for decl in model.decls():
        try:
            value = model[decl]
            if value is not None:
                result[decl.name()] = str(value)
        except Exception:
            result[decl.name()] = "error"
    
    return result


# Singleton instance
error_reporter = ErrorReporter()


def report_verification_failure(message: str, 
                               location: Optional[SourceLocation] = None,
                               category: ErrorCategory = ErrorCategory.ASSERTION,
                               model: Optional[z3.Model] = None,
                               variables: Optional[Dict[str, Any]] = None) -> VerificationError:
    """Report a verification failure with optional counterexample."""
    error = VerificationError(
        category=category,
        severity=ErrorSeverity.ERROR,
        message=message,
        location=location
    )
    
    if model and variables:
        counterexample = error_reporter.generate_counterexample(model, variables)
        error.counterexample = counterexample
    
    error_reporter.add_error(error)
    
    return error


def print_verification_report(output_format: str = "text"):
    """Print the current verification report."""
    error_reporter.print_report(output_format)
