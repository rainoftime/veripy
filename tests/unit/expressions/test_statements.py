"""
Test cases for statements.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestStatements(unittest.TestCase):
    """Test cases for various statements."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_assignment(self):
        """Test variable assignment."""
        vp.scope('test_assignment')
        
        @verify(requires=['x >= 0'], ensures=['ans == x'])
        def assign(x: int) -> int:
            y = x
            z = y
            return z
        
        vp.verify_all()
    
    def test_multiple_assignment(self):
        """Test multiple assignment."""
        vp.scope('test_multi_assign')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def multi_assign(x: int) -> int:
            a = b = c = x
            return a + b + c
        
        vp.verify_all()
    
    def test_return(self):
        """Test return statement."""
        vp.scope('test_return')
        
        @verify(requires=['x >= 0'], ensures=['ans >= x'])
        def ret(x: int) -> int:
            if x > 0:
                return x
            return 0
        
        vp.verify_all()
    
    def test_assert(self):
        """Test assert statement."""
        vp.scope('test_assert')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def with_assert(x: int) -> int:
            assert x >= 0
            return x
        
        vp.verify_all()
    
    def test_assume(self):
        """Test assume statement - using precondition instead."""
        vp.scope('test_assume')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def with_assume(x: int) -> int:
            return x
        
        vp.verify_all()
    
    def test_pass(self):
        """Test pass statement."""
        vp.scope('test_pass')
        
        @verify(requires=['x >= 0'], ensures=['ans == x'])
        def with_pass(x: int) -> int:
            if x > 0:
                pass
            return x
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
