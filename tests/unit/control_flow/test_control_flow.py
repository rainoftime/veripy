"""
Test cases for controlflow.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestControlFlow(unittest.TestCase):
    """Test cases for control flow structures."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # If/Else Statements
    # =========================================================================
    
    def test_simple_if(self):
        """Test simple if statement."""
        vp.scope('test_simple_if')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def abs_value(x: int) -> int:
            if x < 0:
                return -x
            return x
        
        vp.verify_all()
    
    def test_if_else(self):
        """Test if-else statement."""
        vp.scope('test_if_else')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def max_one(x: int) -> int:
            if x > 1:
                return 1
            else:
                return x
        
        vp.verify_all()
    
    def test_if_elif_else(self):
        """Test if-elif-else statement."""
        vp.scope('test_if_elif_else')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def signum(x: int) -> int:
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0
        
        vp.verify_all()
    
    def test_nested_if(self):
        """Test nested if statements."""
        vp.scope('test_nested_if')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def clamp(x: int, y: int) -> int:
            if x < y:
                if x < 0:
                    return 0
                else:
                    return x
            else:
                if y > 10:
                    return 10
                else:
                    return y
        
        vp.verify_all()
    
    def test_if_with_and_condition(self):
        """Test if with boolean AND condition."""
        vp.scope('test_if_and')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def min_value(x: int, y: int) -> int:
            if x < y and x >= 0:
                return x
            else:
                return y
        
        vp.verify_all()
    
    def test_if_with_or_condition(self):
        """Test if with boolean OR condition."""
        vp.scope('test_if_or')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def positive_or_zero(x: int) -> int:
            if x > 0 or x == 0:
                return x
            else:
                return 0
        
        vp.verify_all()
    
    # =========================================================================
    # While Loops
    # =========================================================================
    
    def test_simple_while(self):
        """Test simple while loop."""
        vp.scope('test_simple_while')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def count_down(n: int) -> int:
            count = 0
            while n > 0:
                count = count + 1
                n = n - 1
            return count
        
        vp.verify_all()
    
    def test_while_with_invariants(self):
        """Test while loop with proper invariants."""
        vp.scope('test_while_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * (i - 1) // 2'])
        def double_loop(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
