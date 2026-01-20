"""
Test cases for searchalgorithms.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestSearchAlgorithms(unittest.TestCase):
    """Test cases for search algorithms with invariants."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Binary Search
    # =========================================================================
    
    def test_binary_search(self):
        """Test binary search with classic invariants."""
        vp.scope('test_binary_search')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'])
        def binary_search(arr: List[int], target: int, n: int) -> int:
            lo = 0
            hi = n
            while lo < hi:
                vp.invariant('lo >= 0')
                vp.invariant('hi <= n')
                vp.invariant('lo <= hi')
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            return -1
        
        vp.verify_all()
    
    def test_binary_search_variant(self):
        """Test binary search variant (lo <= hi, return lo)."""
        vp.scope('test_binary_search_var')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def binary_search_var(arr: List[int], target: int, n: int) -> int:
            lo = 0
            hi = n - 1
            while lo <= hi:
                vp.invariant('lo >= 0')
                vp.invariant('hi < n')
                vp.invariant('lo <= hi + 1')
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
        
        vp.verify_all()
    
    def test_binary_search_invariant_preservation(self):
        """Test binary search invariant preservation."""
        vp.scope('test_binary_invariant')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'])
        def binary_invariant(arr: List[int], target: int, n: int) -> int:
            lo = 0
            hi = n
            while lo < hi:
                vp.invariant('lo >= 0')
                vp.invariant('hi <= n')
                vp.invariant('forall i :: i < lo ==> arr[i] < target')
                vp.invariant('forall i :: i >= hi ==> arr[i] >= target')
                mid = (lo + hi) // 2
                if arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        
        vp.verify_all()
    
    # =========================================================================
    # Linear Search
    # =========================================================================
    
    def test_linear_search(self):
        """Test linear search with counter."""
        vp.scope('test_linear_search')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def linear_search(arr: List[int], target: int, n: int) -> int:
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                if arr[i] == target:
                    return i
                i = i + 1
            return -1
        
        vp.verify_all()
    
    def test_linear_search_count(self):
        """Test linear search counting occurrences."""
        vp.scope('test_linear_count')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def count_occurrences(arr: List[int], target: int, n: int) -> int:
            count = 0
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                vp.invariant('count >= 0')
                vp.invariant('count <= i')
                if arr[i] == target:
                    count = count + 1
                i = i + 1
            return count
        
        vp.verify_all()
    
    # =========================================================================
    # Search with Two Pointers
    # =========================================================================
    
    def test_two_sum_sorted(self):
        """Test two-sum on sorted array with two pointers."""
        vp.scope('test_two_sum_sorted')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def two_sum_sorted(arr: List[int], target: int, n: int) -> int:
            lo = 0
            hi = n - 1
            while lo < hi:
                vp.invariant('lo >= 0')
                vp.invariant('hi < n')
                vp.invariant('lo < hi')
                s = arr[lo] + arr[hi]
                if s == target:
                    return 1
                elif s < target:
                    lo = lo + 1
                else:
                    hi = hi - 1
            return 0
        
        vp.verify_all()
    
    def test_find_pivot(self):
        """Test find pivot point in sorted array."""
        vp.scope('test_find_pivot')
        
        @verify(requires=['n >= 2'], ensures=['ans >= 0'])
        def find_pivot(arr: List[int], n: int) -> int:
            lo = 0
            hi = n - 1
            while lo < hi:
                vp.invariant('lo >= 0')
                vp.invariant('hi < n')
                mid = (lo + hi) // 2
                if arr[mid] > arr[hi]:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
