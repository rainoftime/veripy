"""
Tests for termination checker (unit tests for TerminationChecker class).
"""

import unittest
from veripy.auto_active import TerminationChecker


class TestTerminationChecker(unittest.TestCase):
    """Test cases for termination checking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = TerminationChecker()
    
    def test_first_call_allowed(self):
        """Test that first call to a recursive function is allowed."""
        result, message = self.checker.check_termination(
            func_name="factorial",
            decreases="n",
            args=[5]
        )
        
        self.assertTrue(result)
        self.assertIn("First call", message)

    def test_non_decreasing_measure_rejected(self):
        """Termination requires measure to strictly decrease."""
        # First call records measure
        self.checker.check_termination(func_name="factorial", decreases="n", args=[3])
        # Same measure should fail
        result, message = self.checker.check_termination(
            func_name="factorial",
            decreases="n",
            args=[3]
        )
        self.assertFalse(result)
        self.assertIn("does not decrease", message)
    
    def test_termination_without_decreases_fails(self):
        """Test that termination check fails without decreases clause."""
        result, message = self.checker.check_termination(
            func_name="bad_recursion",
            decreases="",
            args=[5]
        )
        
        self.assertFalse(result)
        self.assertIn("decreases clause", message)


if __name__ == '__main__':
    unittest.main()
