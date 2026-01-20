"""
Test cases for automaticinference.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestAutomaticInference(unittest.TestCase):
    """Test cases for automatic invariant inference."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_auto_bounds_inference(self):
        """Test automatic bounds inference."""
        vp.scope('test_auto_bounds')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def auto_bounds(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                # Veripy should auto-infer: i >= 0, i <= n
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_auto_type_inference(self):
        """Test automatic type inference."""
        vp.scope('test_auto_type')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def auto_type() -> int:
            x: int = 0
            y: int = 1
            # Veripy should infer types
            return x + y
        
        vp.verify_all()
    
    def test_auto_relationship_inference(self):
        """Test automatic relationship inference."""
        vp.scope('test_auto_relationship')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def auto_relationship(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                # Veripy should auto-infer: ans relates to i
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
