"""
Tests for error reporting functionality.
"""

import unittest
import json
from veripy.error.reporter import (
    ErrorReporter,
    ErrorCategory,
    ErrorSeverity,
    SourceLocation,
    Counterexample,
    VerificationError
)


class TestErrorReporter(unittest.TestCase):
    """Test cases for error reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = ErrorReporter()
    
    def test_add_error(self):
        """Test adding errors to the reporter."""
        error = VerificationError(
            category=ErrorCategory.ASSERTION,
            severity=ErrorSeverity.ERROR,
            message="Test assertion failed"
        )
        
        self.reporter.add_error(error)
        
        self.assertEqual(len(self.reporter.errors), 1)
        self.assertEqual(self.reporter.statistics["total_errors"], 1)
    
    def test_add_warning(self):
        """Test adding warnings to the reporter."""
        warning = VerificationError(
            category=ErrorCategory.PRECONDITION,
            severity=ErrorSeverity.WARNING,
            message="Precondition might not hold"
        )
        
        self.reporter.add_warning(warning)
        
        self.assertEqual(len(self.reporter.warnings), 1)
        self.assertEqual(self.reporter.statistics["total_warnings"], 1)
    
    def test_source_location(self):
        """Test source location creation."""
        location = SourceLocation(
            file="test.py",
            line=10,
            column=5
        )
        
        self.assertEqual(str(location), "test.py:10:5")
        
        # Test dict conversion
        location_dict = location.to_dict()
        self.assertEqual(location_dict["file"], "test.py")
        self.assertEqual(location_dict["line"], 10)
    
    def test_counterexample(self):
        """Test counterexample creation and formatting."""
        counterexample = Counterexample(
            values={"x": "5", "y": "3"},
            explanation="x and y have specific values"
        )
        
        formatted = counterexample.format()
        
        self.assertIn("x = 5", formatted)
        self.assertIn("y = 3", formatted)
        self.assertIn("Explanation", formatted)
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = VerificationError(
            category=ErrorCategory.TYPE_ERROR,
            severity=ErrorSeverity.ERROR,
            message="Type mismatch",
            suggestion="Check types"
        )
        
        error_dict = error.to_dict()
        
        self.assertEqual(error_dict["category"], "type_error")
        self.assertEqual(error_dict["severity"], "error")
        self.assertEqual(error_dict["message"], "Type mismatch")
        self.assertEqual(error_dict["suggestion"], "Check types")
    
    def test_format_text_report(self):
        """Test text report formatting."""
        # Add some errors and warnings
        self.reporter.add_error(VerificationError(
            category=ErrorCategory.ASSERTION,
            severity=ErrorSeverity.ERROR,
            message="Assertion failed"
        ))
        self.reporter.add_warning(VerificationError(
            category=ErrorCategory.PRECONDITION,
            severity=ErrorSeverity.WARNING,
            message="Precondition warning"
        ))
        
        report = self.reporter.format_report("text")
        
        self.assertIn("VERIFICATION REPORT", report)
        self.assertIn("DETAILED ERRORS", report)
        self.assertIn("WARNINGS", report)
    
    def test_format_json_report(self):
        """Test JSON report formatting."""
        self.reporter.add_error(VerificationError(
            category=ErrorCategory.ASSERTION,
            severity=ErrorSeverity.ERROR,
            message="Assertion failed"
        ))
        
        report = self.reporter.format_report("json")
        
        # Should be valid JSON
        report_dict = json.loads(report)
        
        self.assertIn("summary", report_dict)
        self.assertIn("errors", report_dict)
        self.assertEqual(report_dict["passed"], False)


if __name__ == '__main__':
    unittest.main()
