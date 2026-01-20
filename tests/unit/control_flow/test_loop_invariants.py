"""
Test cases for loopinvariants.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestLoopInvariants(unittest.TestCase):
    """Test cases for loop invariants."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Manual Invariants
    # =========================================================================
    
    def test_basic_invariant(self):
        """Test basic loop invariant."""
        vp.scope('test_basic_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def basic_loop(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_accumulator_invariant(self):
        """Test loop with accumulator invariant."""
        vp.scope('test_accumulator_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * (n - 1) // 2'])
        def sum_to_n(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans == i * (i - 1) // 2')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_multiple_invariants(self):
        """Test loop with multiple invariants."""
        vp.scope('test_multiple_invariants')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def product(x: int, y: int) -> int:
            ans = 0
            i = 0
            j = 0
            while i < x:
                invariant('i >= 0')
                invariant('i <= x')
                invariant('ans == i * y')
                j = 0
                while j < y:
                    invariant('j >= 0')
                    invariant('j <= y')
                    invariant('ans == i * y + j')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_invariant_with_comparison(self):
        """Test invariant with comparison."""
        vp.scope('test_invariant_comparison')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def find_max(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            max_val = arr[0]
            i = 1
            while i < n:
                invariant('i >= 1')
                invariant('i <= n')
                invariant('max_val >= arr[0]')
                invariant('forall j: int :: j >= 0 and j < i ==> max_val >= arr[j]')
                if arr[i] > max_val:
                    max_val = arr[i]
                i = i + 1
            return max_val
        
        vp.verify_all()
    
    def test_invariant_preservation(self):
        """Test invariant preservation."""
        vp.scope('test_invariant_preservation')
        
        @verify(requires=['n >= 0'], ensures=['ans >= n'])
        def sum_greater_than_n(n: int) -> int:
            ans = n
            i = 0
            while i < n:
                invariant('ans >= n')
                invariant('i >= 0')
                invariant('i <= n')
                ans = ans + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    # =========================================================================
    # Invariant Types
    # =========================================================================
    
    def test_bounds_invariant(self):
        """Test bounds invariant."""
        vp.scope('test_bounds_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bounded_loop(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_type_invariant(self):
        """Test type invariant."""
        vp.scope('test_type_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def typed_loop(n: int) -> int:
            i: int = 0
            ans: int = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_relationship_invariant(self):
        """Test relationship between loop variables."""
        vp.scope('test_relationship_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * n'])
        def square_loop(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans == i * n')
                j = 0
                while j < n:
                    invariant('j >= 0')
                    invariant('j <= n')
                    invariant('ans == i * n + j')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    # =========================================================================
    # Complex Loop Patterns
    # =========================================================================
    
    def test_nested_loop_invariants(self):
        """Test nested loop with invariants."""
        vp.scope('test_nested_loop_invariants')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans == n * m'])
        def matrix_sum(n: int, m: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                j = 0
                while j < m:
                    invariant('j >= 0')
                    invariant('j <= m')
                    invariant('ans == i * m + j')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_loop_with_break_invariants(self):
        """Test loop with break and invariants."""
        vp.scope('test_break_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans >= -1'])
        def find_value(arr: List[int], n: int, target: int) -> int:
            i = 0
            ans = -1
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                if arr[i] == target:
                    ans = i
                    break
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_loop_with_continue_invariants(self):
        """Test loop with continue and invariants."""
        vp.scope('test_continue_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def skip_odds(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                i = i + 1
                if i % 2 == 1:
                    continue
                ans = ans + i
            return ans
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
