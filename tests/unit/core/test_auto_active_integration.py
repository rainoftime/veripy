"""
Integration tests for auto-active verification features.
"""

import unittest
import veripy as vp
from veripy import verify
from veripy.error.reporter import (
    ErrorReporter,
    ErrorCategory,
    ErrorSeverity,
    SourceLocation,
    VerificationError
)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features."""
    
    def setUp(self):
        """Set up verification for each test."""
        vp.enable_verification()
    
    def test_verify_with_auto_invariants(self):
        """Test verification using auto-inferred invariants."""
        vp.scope('test_auto_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def sum_to_n(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                ans = ans + i
                i = i + 1
            return ans
        
        # This should work (though may need manual invariants)
        vp.verify_all()
    
    def test_error_reporting_integration(self):
        """Test error reporting integration."""
        reporter = ErrorReporter()
        
        # Simulate a verification failure
        reporter.add_error(VerificationError(
            category=ErrorCategory.ASSERTION,
            severity=ErrorSeverity.ERROR,
            message="Precondition does not imply weakest precondition",
            location=SourceLocation("test.py", 10, 5),
            suggestion="Add more specific preconditions"
        ))
        
        # Generate report
        report = reporter.format_report("text")
        
        self.assertIn("VERIFICATION REPORT", report)
        self.assertIn("Suggestion", report)


if __name__ == '__main__':
    unittest.main()
