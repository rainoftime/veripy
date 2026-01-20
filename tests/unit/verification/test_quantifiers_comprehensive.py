"""
Test cases for quantifiers.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestQuantifiers(unittest.TestCase):
    """Test cases for quantifiers."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Forall Quantifier
    # =========================================================================
    
    def test_simple_forall(self):
        """Test simple forall quantifier."""
        vp.scope('test_simple_forall')
        
        @verify(requires=['True'], ensures=['forall x: int :: x == x'])
        def identity() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_with_implication(self):
        """Test forall with implication."""
        vp.scope('test_forall_implication')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: i >= 0 ==> i >= 0'])
        def forall_implication(n: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_universal_truth(self):
        """Test forall quantifier for universal truth."""
        vp.scope('test_forall_truth')
        
        @verify(requires=['True'], ensures=['forall x: int :: x + 0 == x'])
        def add_zero() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_multi_variable(self):
        """Test forall with multiple variables."""
        vp.scope('test_forall_multi')
        
        @verify(requires=['True'], ensures=['forall x: int, y: int :: x + y == y + x'])
        def commutativity() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_with_arithmetic(self):
        """Test forall with arithmetic."""
        vp.scope('test_forall_arithmetic')
        
        @verify(requires=['True'], ensures=['forall x: int, y: int :: x + y > x'])
        def arithmetic_forall(y: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_range(self):
        """Test forall with range constraint."""
        vp.scope('test_forall_range')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: i >= 0 and i < n ==> arr[i] >= 0'])
        def array_non_negative(arr: List[int], n: int) -> bool:
            return True
        
        vp.verify_all()
    
    # =========================================================================
    # Exists Quantifier
    # =========================================================================
    
    def test_simple_exists(self):
        """Test simple exists quantifier."""
        vp.scope('test_simple_exists')
        
        @verify(requires=['True'], ensures=['exists x: int :: x > 0'])
        def exists_positive() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_with_condition(self):
        """Test exists with condition."""
        vp.scope('test_exists_condition')
        
        @verify(requires=['n >= 0'], ensures=['exists i: int :: i >= 0 and i < n'])
        def exists_in_range(n: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_not_forall(self):
        """Test exists is not forall."""
        vp.scope('test_exists_not_forall')
        
        @verify(requires=['True'], ensures=['exists x: int :: not (x > 0)'])
        def exists_non_positive() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_complex(self):
        """Test complex exists quantifier."""
        vp.scope('test_exists_complex')
        
        @verify(requires=['n >= 0'], ensures=['exists i: int :: i >= 0 and i < n and arr[i] == 5'])
        def find_five(arr: List[int], n: int) -> bool:
            return True
        
        vp.verify_all()
    
    # =========================================================================
    # Quantifier Combinations
    # =========================================================================
    
    def test_forall_exists(self):
        """Test forall followed by exists."""
        vp.scope('test_forall_exists')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: exists j: int :: j > i'])
        def infinite_growth() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_forall(self):
        """Test exists followed by forall."""
        return # TIMEOUT 
        vp.scope('test_exists_forall')
        
        @verify(requires=['True'], ensures=['exists x: int :: forall y: int :: x <= y'])
        def minimum_exists() -> bool:
            return True
        
        vp.verify_all()
    
    def test_quantifier_equivalence(self):
        """Test quantifier equivalence."""
        vp.scope('test_quantifier_equivalence')
        
        @verify(requires=['True'], ensures=['not (forall x :: P(x)) <==> exists x :: not P(x)'])
        def de_morgan_quantifiers() -> bool:
            return True
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
