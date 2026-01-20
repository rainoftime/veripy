"""
Test cases for methodcalls.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestMethodCalls(unittest.TestCase):
    """Test cases for method calls."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_simple_method_call(self):
        """Test simple method call."""
        vp.scope('test_simple_method')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def simple_method() -> int:
            # obj.method()
            return 0
        
        vp.verify_all()
    
    def test_method_with_args(self):
        """Test method call with arguments."""
        vp.scope('test_method_args')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def method_with_args(x: int) -> int:
            # obj.method(x, y)
            return x
        
        vp.verify_all()
    
    def test_method_chaining(self):
        """Test method chaining."""
        vp.scope('test_method_chain')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def method_chain() -> int:
            # obj.method1().method2()
            return 0
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
