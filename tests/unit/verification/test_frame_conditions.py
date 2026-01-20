"""
Test cases for frameconditions.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestFrameConditions(unittest.TestCase):
    """Test cases for frame conditions and modifies clauses."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_single_variable_modifies(self):
        """Test modifies clause with single variable."""
        vp.scope('test_single_modifies')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def increment(x: int) -> int:
            return x + 1
        
        vp.verify_all()
    
    def test_multiple_variables(self):
        """Test with multiple variables."""
        vp.scope('test_multi_variables')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def swap(x: int, y: int) -> int:
            temp = x
            x = y
            y = temp
            return x + y
        
        vp.verify_all()
    
    def test_array_modifies(self):
        """Test array modification frame."""
        vp.scope('test_array_modifies')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def update_array(arr: List[int], n: int) -> int:
            arr[0] = n
            return arr[0]
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
