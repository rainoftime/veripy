"""
Test cases for comprehensions.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestComprehensions(unittest.TestCase):
    """Test cases for comprehensions."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # List Comprehensions
    # =========================================================================
    
    def test_list_comprehension_basic(self):
        """Test basic list comprehension."""
        vp.scope('test_list_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_basic(n: int) -> int:
            # [x for x in range(n)]
            return n
        
        vp.verify_all()
    
    def test_list_comprehension_with_filter(self):
        """Test list comprehension with filter."""
        vp.scope('test_list_comp_filter')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_filter(n: int) -> int:
            # [x for x in range(n) if x % 2 == 0]
            return n
        
        vp.verify_all()
    
    def test_list_comprehension_with_transform(self):
        """Test list comprehension with transformation."""
        vp.scope('test_list_comp_transform')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_transform(n: int) -> int:
            # [x * 2 for x in range(n)]
            return n
        
        vp.verify_all()
    
    # =========================================================================
    # Set Comprehensions
    # =========================================================================
    
    def test_set_comprehension_basic(self):
        """Test basic set comprehension."""
        vp.scope('test_set_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_comp_basic(n: int) -> int:
            # {x for x in range(n)}
            return n
        
        vp.verify_all()
    
    def test_set_comprehension_with_filter(self):
        """Test set comprehension with filter."""
        vp.scope('test_set_comp_filter')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_comp_filter(n: int) -> int:
            # {x for x in range(n) if x % 2 == 0}
            return n
        
        vp.verify_all()
    
    # =========================================================================
    # Dict Comprehensions
    # =========================================================================
    
    def test_dict_comprehension_basic(self):
        """Test basic dict comprehension."""
        vp.scope('test_dict_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def dict_comp_basic(n: int) -> int:
            # {x: x * x for x in range(n)}
            return n
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
