"""
Test cases for edgecases.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_zero_values(self):
        """Test with zero values."""
        vp.scope('test_zero')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def zero_test(x: int) -> int:
            if x == 0:
                return 1
            return x
        
        vp.verify_all()
    
    def test_negative_values(self):
        """Test with negative values."""
        vp.scope('test_negative')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def neg_test(x: int) -> int:
            if x < 0:
                return -x
            return x
        
        vp.verify_all()
    
    def test_large_values(self):
        """Test with large values."""
        vp.scope('test_large')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def large_test() -> int:
            large = 10**10
            return large
        
        vp.verify_all()
    
    def test_empty_list(self):
        """Test with empty list."""
        vp.scope('test_empty_list')
        
        @verify(requires=['True'], ensures=['ans == 0'])
        def empty_test(arr: List[int]) -> int:
            return 0
        
        vp.verify_all()
    
    def test_single_element(self):
        """Test with single element."""
        vp.scope('test_single')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def single_test(x: int) -> int:
            arr = [x]
            return arr[0]
        
        vp.verify_all()
    
    def test_identity_operations(self):
        """Test identity operations (x - x = 0, x / x = 1)."""
        vp.scope('test_identity')
        
        @verify(requires=['x > 0'], ensures=['ans >= 0'])
        def identity_test(x: int) -> int:
            diff = x - x
            # quot = x // x  # potential division by zero
            return diff
        
        vp.verify_all()
    
    def test_constant_folding(self):
        """Test constant folding."""
        vp.scope('test_const_fold')
        
        @verify(requires=['True'], ensures=['ans == 42'])
        def const_fold() -> int:
            return 40 + 2
        
        vp.verify_all()


if __name__ == '__main__':
    unittest.main()


if __name__ == "__main__":
    unittest.main()
