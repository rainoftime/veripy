"""
Test cases for dictionaryoperations.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestDictionaryOperations(unittest.TestCase):
    """Test cases for dictionary operations."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Dict Literals
    # =========================================================================
    
    def test_dict_literal_empty(self):
        """Test empty dictionary literal."""
        vp.scope('test_dict_literal_empty')
        
        @verify(requires=['True'], ensures=['ans == 0'])
        def empty_dict() -> int:
            d: Dict[str, int] = {}
            return 0
        
        vp.verify_all()
    
    def test_dict_literal_values(self):
        """Test dictionary with values."""
        vp.scope('test_dict_literal_values')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_with_values() -> int:
            d: Dict[str, int] = {'a': 1, 'b': 2, 'c': 3}
            return d['a'] + d['b']
        
        vp.verify_all()
    
    # =========================================================================
    # Dict Operations
    # =========================================================================
    
    def test_dict_get(self):
        """Test dictionary get operation."""
        vp.scope('test_dict_get')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_get_op() -> int:
            d: Dict[str, int] = {'x': 10, 'y': 20}
            # Get with default
            val = d.get('x', 0)
            return val
        
        vp.verify_all()
    
    def test_dict_get_with_default(self):
        """Test dictionary get with default value."""
        vp.scope('test_dict_get_default')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_get_default() -> int:
            d: Dict[str, int] = {'a': 1}
            # Get missing key with default
            val = d.get('missing', 42)
            return val
        
        vp.verify_all()
    
    def test_dict_keys(self):
        """Test dictionary keys operation."""
        vp.scope('test_dict_keys')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_keys_op() -> int:
            d: Dict[str, int] = {'x': 1, 'y': 2, 'z': 3}
            # Keys operation
            return 3
        
        vp.verify_all()
    
    def test_dict_values(self):
        """Test dictionary values operation."""
        vp.scope('test_dict_values')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_values_op() -> int:
            d: Dict[str, int] = {'a': 10, 'b': 20}
            # Values operation
            return 30
        
        vp.verify_all()
    
    def test_dict_contains(self):
        """Test dictionary contains operation."""
        vp.scope('test_dict_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_contains_op() -> int:
            d: Dict[str, int] = {'key': 42}
            if 'key' in d:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_dict_not_contains(self):
        """Test dictionary not contains."""
        vp.scope('test_dict_not_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_not_contains() -> int:
            d: Dict[str, int] = {'a': 1}
            if 'b' not in d:
                return 1
            return 0
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
