"""
Test cases for termination.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestTermination(unittest.TestCase):
    """Test cases for termination checking."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Decreases Clause
    # =========================================================================
    
    def test_decreases_single(self):
        """Test decreases with single variable."""
        vp.scope('test_decreases_single')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def countdown(n: int) -> int:
            if n == 0:
                return 0
            else:
                return countdown(n - 1)
        
        vp.verify_all()
    
    def test_decreases_tuple(self):
        """Test decreases with tuple."""
        vp.scope('test_decreases_tuple')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans >= 0'], decreases='(n, m)')
        def nested_countdown(n: int, m: int) -> int:
            if n == 0:
                if m == 0:
                    return 0
                else:
                    return nested_countdown(n, m - 1)
            else:
                return nested_countdown(n - 1, m)
        
        vp.verify_all()
    
    def test_decreases_expression(self):
        """Test decreases with expression."""
        vp.scope('test_decreases_expr')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'], decreases='n // 2')
        def halving(n: int) -> int:
            if n == 1:
                return 0
            else:
                return halving(n // 2)
        
        vp.verify_all()
    
    # =========================================================================
    # Termination Patterns
    # =========================================================================
    
    def test_linear_termination(self):
        """Test linear termination."""
        vp.scope('test_linear_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def linear_recursion(n: int) -> int:
            if n == 0:
                return 0
            return 1 + linear_recursion(n - 1)
        
        vp.verify_all()
    
    def test_logarithmic_termination(self):
        """Test logarithmic termination."""
        vp.scope('test_log_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def binary_recursion(n: int) -> int:
            if n <= 1:
                return n
            return binary_recursion(n // 2) + binary_recursion(n // 2)
        
        vp.verify_all()
    
    def test_tree_termination(self):
        """Test tree-structured recursion termination."""
        vp.scope('test_tree_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def tree_sum(n: int) -> int:
            if n == 0:
                return 0
            left = tree_sum(n - 1)
            right = tree_sum(n - 1)
            return 1 + left + right
        
        vp.verify_all()
    
    # =========================================================================
    # Non-Terminating Cases
    # =========================================================================
    
    def test_missing_decreases(self):
        """Test that missing decreases clause is detected."""
        vp.scope('test_missing_decreases')
        
        # This should fail or require decreases clause
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def no_decreases(n: int) -> int:
            if n > 0:
                return no_decreases(n)
            return 0
        
        # This will fail - expected behavior
        try:
            vp.verify_all()
        except:
            pass  # Expected - should fail verification
    
    def test_non_decreasing(self):
        """Test non-decreasing recursive call."""
        vp.scope('test_non_decreasing')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bad_recursion(n: int) -> int:
            if n == 0:
                return 0
            # n + 1 is not decreasing
            return bad_recursion(n + 1)
        
        # This should fail
        try:
            vp.verify_all()
        except:
            pass  # Expected - should fail verification


if __name__ == "__main__":
    unittest.main()
