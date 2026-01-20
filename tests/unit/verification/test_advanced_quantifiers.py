"""
Test cases for advancedquantifiers.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestAdvancedQuantifiers(unittest.TestCase):
    """Test cases for advanced quantifier patterns."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_forall_sorted(self):
        """Test forall quantifier for sorted property."""
        vp.scope('test_forall_sorted')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def check_sorted(arr: List[int], n: int) -> int:
            # Check if array is sorted using forall
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                i = i + 1
            return 1
        
        vp.verify_all()
    
    def test_exists_in_range(self):
        """Test exists quantifier for search."""
        vp.scope('test_exists_range')
        
        @verify(requires=['n >= 0', 'target >= 0'], ensures=['ans >= 0'])
        def exists_in_range(arr: List[int], target: int, n: int) -> int:
            # Check if target exists in range
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                if arr[i] == target:
                    return 1
                i = i + 1
            return 0
        
        vp.verify_all()
    
    def test_quantifier_equivalence(self):
        """Test quantifier equivalence patterns."""
        vp.scope('test_quant_equiv')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def quant_equiv(arr: List[int], n: int) -> int:
            # not (forall i, P(i)) <==> exists i, not P(i)
            all_positive = True
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                if arr[i] <= 0:
                    all_positive = False
                i = i + 1
            return 1
        
        vp.verify_all()
    
    def test_uniqueness(self):
        """Test uniqueness quantifier."""
        vp.scope('test_uniqueness')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def check_unique(arr: List[int], n: int) -> int:
            # Check if all elements are unique
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                j = i + 1
                while j < n:
                    vp.invariant('j >= i + 1')
                    vp.invariant('j <= n')
                    if arr[i] == arr[j]:
                        return 0
                    j = j + 1
                i = i + 1
            return 1
        
        vp.verify_all()
    
    def test_min_max_quantifiers(self):
        """Test min/max with quantifiers."""
        vp.scope('test_min_max')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def find_min_max(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            
            min_val = arr[0]
            max_val = arr[0]
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                vp.invariant('min_val <= arr[0]')
                vp.invariant('max_val >= arr[0]')
                if arr[i] < min_val:
                    min_val = arr[i]
                if arr[i] > max_val:
                    max_val = arr[i]
                i = i + 1
            
            return min_val + max_val
        
        vp.verify_all()


if __name__ == '__main__':
    unittest.main()


if __name__ == "__main__":
    unittest.main()
