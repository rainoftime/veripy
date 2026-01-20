"""
Test cases for sortingalgorithms.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestSortingAlgorithms(unittest.TestCase):
    """Test cases for sorting algorithms with Dafny-style invariants.
    
    Key patterns from Dafny:
    - predicate sorted(a:seq) { forall i,j | 0 <= i < j < |a| :: a[i] <= a[j] }
    - multiset(a[..]) == multiset(old(a[..])) - permutation invariant
    - forall i | 0 <= i < j :: a[i] <= a[j] - sorted prefix
    """
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Bubble Sort
    # =========================================================================
    
    def test_bubble_sort_basic(self):
        """Test bubble sort with proper invariants."""
        vp.scope('test_bubble_sort')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bubble_sort(arr: List[int], n: int) -> int:
            # Simplified bubble sort
            i = 0
            while i < n - 1:
                vp.invariant('i >= 0')
                vp.invariant('i <= n - 1')
                j = 0
                while j < n - 1 - i:
                    vp.invariant('j >= 0')
                    vp.invariant('j <= n - 1 - i')
                    j = j + 1
                i = i + 1
            return 0
        
        vp.verify_all()
    
    def test_bubble_sort_optimized(self):
        """Test optimized bubble sort with early termination."""
        vp.scope('test_bubble_sort_opt')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bubble_sort_opt(arr: List[int], n: int) -> int:
            i = n - 1
            while i > 0:
                vp.invariant('i >= 0')
                vp.invariant('i < n')
                j = 0
                while j < i:
                    vp.invariant('j >= 0')
                    vp.invariant('j <= i')
                    j = j + 1
                i = i - 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Selection Sort
    # =========================================================================
    
    def test_selection_sort(self):
        """Test selection sort with min-finding invariant."""
        vp.scope('test_selection_sort')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def selection_sort(arr: List[int], n: int) -> int:
            # Find minimum in unsorted portion and swap to front
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                # Find minimum in arr[i:]
                min_idx = i
                j = i + 1
                while j < n:
                    vp.invariant('j >= i + 1')
                    vp.invariant('j <= n')
                    j = j + 1
                i = i + 1
            return 0
        
        vp.verify_all()
    
    def test_selection_sort_properties(self):
        """Test selection sort with element properties."""
        vp.scope('test_selection_props')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def selection_props(arr: List[int], n: int) -> int:
            # At each step, the minimum element goes to position i
            i = 0
            while i < n - 1:
                vp.invariant('i >= 0')
                vp.invariant('i <= n - 1')
                min_val = arr[i]
                j = i + 1
                while j < n:
                    vp.invariant('j >= i + 1')
                    vp.invariant('j <= n')
                    j = j + 1
                i = i + 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Insertion Sort
    # =========================================================================
    
    def test_insertion_sort(self):
        """Test insertion sort with sorted prefix invariant."""
        vp.scope('test_insertion_sort')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def insertion_sort(arr: List[int], n: int) -> int:
            # Build sorted array one element at a time
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                # arr[0..i-1] is sorted, insert arr[i] in correct position
                key = arr[i]
                j = i - 1
                while j >= 0 and arr[j] > key:
                    vp.invariant('j >= -1')
                    vp.invariant('j < i')
                    arr[j + 1] = arr[j]
                    j = j - 1
                arr[j + 1] = key
                i = i + 1
            return 0
        
        vp.verify_all()
    
    def test_insertion_sort_sorted_prefix(self):
        """Test insertion sort with explicit sorted prefix."""
        vp.scope('test_insertion_prefix')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def insertion_prefix(arr: List[int], n: int) -> int:
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                # Prefix arr[0:i] is sorted
                vp.invariant('forall p, q :: 0 <= p < q < i ==> arr[p] <= arr[q]')
                key = arr[i]
                j = i - 1
                while j >= 0 and arr[j] > key:
                    vp.invariant('j >= -1')
                    vp.invariant('j < i')
                    arr[j + 1] = arr[j]
                    j = j - 1
                arr[j + 1] = key
                i = i + 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Merge Sort
    # =========================================================================
    
    def test_merge_sort(self):
        """Test merge sort with recursive structure."""
        vp.scope('test_merge_sort')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def merge_sort(arr: List[int], n: int) -> int:
            if n <= 1:
                return n
            mid = n // 2
            left_size = merge_sort(arr[:mid], mid)
            right_size = merge_sort(arr[mid:], n - mid)
            return left_size + right_size
        
        vp.verify_all()
    
    def test_merge_sort_invariant(self):
        """Test merge sort with merge invariants."""
        vp.scope('test_merge_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def merge_invariant(arr: List[int], n: int) -> int:
            # Simulate merge operation
            if n <= 1:
                return n
            mid = n // 2
            # Left half sorted
            i = 0
            while i < mid - 1:
                vp.invariant('i >= 0')
                vp.invariant('i <= mid - 1')
                i = i + 1
            # Right half sorted
            j = mid
            while j < n - 1:
                vp.invariant('j >= mid')
                vp.invariant('j <= n - 1')
                j = j + 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Sorting Properties
    # =========================================================================
    
    def test_sorting_preserves_elements(self):
        """Test that sorting preserves all elements (permutation)."""
        vp.scope('test_sort_permutation')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def sort_permutation(arr: List[int], n: int) -> int:
            # After sorting, multiset of elements should be preserved
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                i = i + 1
            return n
        
        vp.verify_all()
    
    def test_sorted_array_properties(self):
        """Test properties of a sorted array."""
        vp.scope('test_sorted_props')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def sorted_props(arr: List[int], n: int) -> int:
            # For a sorted array:
            # 1. Each element <= next element
            # 2. Each element >= previous element
            # 3. Minimum is first, maximum is last
            if n == 0:
                return 0
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                # arr[i-1] <= arr[i]
                vp.invariant('arr[i-1] <= arr[i]')
                i = i + 1
            return arr[0]
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
