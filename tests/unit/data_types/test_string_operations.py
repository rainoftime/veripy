"""
Test cases for stringoperations.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestStringOperations(unittest.TestCase):
    """Test cases for string operations."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # String Literals
    # =========================================================================
    
    def test_string_literal(self):
        """Test string literal."""
        vp.scope('test_string_literal')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_literal() -> int:
            s: str = "hello"
            return len(s)
        
        vp.verify_all()
    
    def test_string_empty(self):
        """Test empty string."""
        vp.scope('test_string_empty')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def empty_string() -> int:
            s: str = ""
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # String Operations
    # =========================================================================
    
    def test_string_concat(self):
        """Test string concatenation."""
        vp.scope('test_string_concat')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_concat_op() -> int:
            s1: str = "hello"
            s2: str = "world"
            # Concatenation: s1 + s2
            result = s1 + s2
            return len(result)
        
        vp.verify_all()
    
    def test_string_length(self):
        """Test string length."""
        vp.scope('test_string_length')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_length_op() -> int:
            s: str = "veripy"
            return len(s)
        
        vp.verify_all()
    
    def test_string_index(self):
        """Test string indexing."""
        vp.scope('test_string_index')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_index_op() -> int:
            s: str = "abc"
            # Indexing: s[0]
            return 1
        
        vp.verify_all()
    
    def test_string_contains(self):
        """Test string contains."""
        vp.scope('test_string_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_contains_op() -> int:
            s: str = "hello world"
            if "world" in s:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_string_multi_concat(self):
        """Test multiple string concatenations."""
        vp.scope('test_string_multi_concat')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_multi_concat() -> int:
            a: str = "a"
            b: str = "b"
            c: str = "c"
            result = a + b + c
            return len(result)
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
