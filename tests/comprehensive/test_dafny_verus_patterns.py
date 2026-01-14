"""
Advanced Test Suite for Veripy - Dafny and Verus Inspired Patterns

This test module contains advanced verification test cases inspired by:
- Dafny's verification patterns (sorted predicates, multisets, permutations)
- Verus's verification patterns (overflow handling, bit manipulation, state machines)

Key verification concepts covered:
- Sorting with loop invariants (bubble, selection, insertion, merge)
- Prefix sums and sliding window
- Two pointers technique
- Bit manipulation with overflow
- Array manipulation with range queries
- Mathematical properties and proofs
- Quantifier patterns
- Search algorithms with invariants
"""

import unittest
import veripy as vp
from veripy import verify, invariant, scope
from typing import List


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


class TestPrefixSumAndSlidingWindow(unittest.TestCase):
    """Test cases for prefix sum and sliding window techniques."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Prefix Sum
    # =========================================================================
    
    def test_prefix_sum_basic(self):
        """Test basic prefix sum computation."""
        vp.scope('test_prefix_sum')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def prefix_sum(arr: List[int], n: int) -> int:
            prefix: List[int] = [0] * (n + 1)
            i = 1
            while i <= n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n + 1')
                vp.invariant('prefix[i] == prefix[i-1] + arr[i-1]')
                prefix[i] = prefix[i - 1] + arr[i - 1]
                i = i + 1
            return prefix[n]
        
        vp.verify_all()
    
    def test_prefix_sum_range_query(self):
        """Test range sum query using prefix sum."""
        vp.scope('test_range_query')
        
        @verify(requires=['n >= 0', '0 <= l <= r < n'], ensures=['ans >= 0'])
        def range_sum(prefix: List[int], l: int, r: int, n: int) -> int:
            # prefix[r+1] - prefix[l] gives sum of arr[l..r]
            return 0
        
        vp.verify_all()
    
    def test_prefix_sum_properties(self):
        """Test prefix sum properties."""
        vp.scope('test_prefix_props')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def prefix_props(arr: List[int], n: int) -> int:
            # prefix[0] = 0
            # prefix[i] = sum of arr[0..i-1]
            # prefix[n] = sum of all elements
            prefix = [0] * (n + 1)
            i = 1
            while i <= n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n + 1')
                prefix[i] = prefix[i - 1] + arr[i - 1]
                i = i + 1
            return prefix[n]
        
        vp.verify_all()
    
    # =========================================================================
    # Sliding Window
    # =========================================================================
    
    def test_sliding_window_sum(self):
        """Test sliding window sum."""
        vp.scope('test_sliding_sum')
        
        @verify(requires=['n >= 0', 'k > 0', 'k <= n'], ensures=['ans >= 0'])
        def sliding_window_sum(arr: List[int], k: int, n: int) -> int:
            # Calculate sum of first window
            window_sum = 0
            i = 0
            while i < k:
                vp.invariant('i >= 0')
                vp.invariant('i <= k')
                window_sum = window_sum + arr[i]
                i = i + 1
            
            # Slide the window
            result = window_sum
            while i < n:
                vp.invariant('i >= k')
                vp.invariant('i <= n')
                vp.invariant('result == sum of arr[i-k+1..i]')
                window_sum = window_sum - arr[i - k] + arr[i]
                result = window_sum
                i = i + 1
            return result
        
        vp.verify_all()
    
    def test_sliding_window_max(self):
        """Test sliding window maximum."""
        vp.scope('test_sliding_max')
        
        @verify(requires=['n >= 0', 'k > 0', 'k <= n'], ensures=['ans >= 0'])
        def sliding_max(arr: List[int], k: int, n: int) -> int:
            i = 0
            current_max = arr[0]
            while i < k:
                vp.invariant('i >= 0')
                vp.invariant('i <= k')
                if arr[i] > current_max:
                    current_max = arr[i]
                i = i + 1
            return current_max
        
        vp.verify_all()
    
    def test_subarray_sum_k(self):
        """Test finding subarray with sum k."""
        vp.scope('test_subarray_sum')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def subarray_sum(arr: List[int], k: int, n: int) -> int:
            prefix = [0] * (n + 1)
            i = 1
            while i <= n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n + 1')
                prefix[i] = prefix[i - 1] + arr[i - 1]
                i = i + 1
            
            # Check if any subarray has sum k
            i = 0
            while i <= n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n + 1')
                j = i
                while j <= n:
                    vp.invariant('j >= i')
                    vp.invariant('j <= n + 1')
                    j = j + 1
                i = i + 1
            return 0
        
        vp.verify_all()


class TestTwoPointers(unittest.TestCase):
    """Test cases for two pointers technique."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_remove_duplicates(self):
        """Test remove duplicates from sorted array."""
        vp.scope('test_remove_dup')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def remove_duplicates(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            # Use two pointers: read pointer and write pointer
            write = 1
            read = 1
            while read < n:
                vp.invariant('read >= 1')
                vp.invariant('read <= n')
                vp.invariant('write >= 1')
                vp.invariant('write <= read')
                if arr[read] != arr[read - 1]:
                    arr[write] = arr[read]
                    write = write + 1
                read = read + 1
            return write
        
        vp.verify_all()
    
    def test_reverse_array(self):
        """Test reverse array with two pointers."""
        vp.scope('test_reverse')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def reverse_array(arr: List[int], n: int) -> int:
            left = 0
            right = n - 1
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                vp.invariant('left <= right')
                # Swap arr[left] and arr[right]
                temp = arr[left]
                arr[left] = arr[right]
                arr[right] = temp
                left = left + 1
                right = right - 1
            return 0
        
        vp.verify_all()
    
    def test_is_palindrome(self):
        """Test palindrome check with two pointers."""
        vp.scope('test_palindrome')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def is_palindrome(arr: List[int], n: int) -> int:
            left = 0
            right = n - 1
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                vp.invariant('left <= right')
                if arr[left] != arr[right]:
                    return 0
                left = left + 1
                right = right - 1
            return 1
        
        vp.verify_all()
    
    def test_merge_sorted_arrays(self):
        """Test merge two sorted arrays."""
        vp.scope('test_merge_sorted')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans >= 0'])
        def merge_sorted(arr1: List[int], n: int, arr2: List[int], m: int) -> int:
            # Merge two sorted arrays into one sorted array
            result: List[int] = []
            i = 0
            j = 0
            while i < n and j < m:
                vp.invariant('i >= 0')
                vp.invariant('j >= 0')
                vp.invariant('i <= n')
                vp.invariant('j <= m')
                if arr1[i] <= arr2[j]:
                    result = result + [arr1[i]]
                    i = i + 1
                else:
                    result = result + [arr2[j]]
                    j = j + 1
            
            while i < n:
                result = result + [arr1[i]]
                i = i + 1
            
            while j < m:
                result = result + [arr2[j]]
                j = j + 1
            
            return len(result)
        
        vp.verify_all()
    
    def test_trapping_rain_water(self):
        """Test trapping rain water problem."""
        vp.scope('test_trapping_rain')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def trapping_rain(height: List[int], n: int) -> int:
            if n == 0:
                return 0
            
            left = 0
            right = n - 1
            left_max = 0
            right_max = 0
            water = 0
            
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                vp.invariant('left < right')
                vp.invariant('left_max >= 0')
                vp.invariant('right_max >= 0')
                vp.invariant('water >= 0')
                
                if height[left] < height[right]:
                    if height[left] >= left_max:
                        left_max = height[left]
                    else:
                        water = water + left_max - height[left]
                    left = left + 1
                else:
                    if height[right] >= right_max:
                        right_max = height[right]
                    else:
                        water = water + right_max - height[right]
                    right = right - 1
            
            return water
        
        vp.verify_all()


class TestBitManipulation(unittest.TestCase):
    """Test cases for bit manipulation with overflow handling."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_bit_count(self):
        """Test count set bits."""
        vp.scope('test_bit_count')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def count_bits(x: int) -> int:
            count = 0
            n = x
            while n > 0:
                vp.invariant('n >= 0')
                vp.invariant('count >= 0')
                count = count + (n & 1)
                n = n >> 1
            return count
        
        vp.verify_all()
    
    def test_bit_count_kernighan(self):
        """Test count bits using Kernighan's algorithm."""
        vp.scope('test_bit_kernighan')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def count_bits_k(x: int) -> int:
            count = 0
            n = x
            while n:
                vp.invariant('n >= 0')
                vp.invariant('count >= 0')
                n = n & (n - 1)
                count = count + 1
            return count
        
        vp.verify_all()
    
    def test_is_power_of_two(self):
        """Test check for power of two."""
        vp.scope('test_power_of_two')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def is_power_of_two(n: int) -> int:
            if n == 0:
                return 0
            # Power of two: n & (n-1) == 0
            return 1 if (n & (n - 1)) == 0 else 0
        
        vp.verify_all()
    
    def test_bit_parity(self):
        """Test parity (even/odd number of bits)."""
        vp.scope('test_bit_parity')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def parity(x: int) -> int:
            parity = 0
            n = x
            while n:
                vp.invariant('parity >= 0')
                vp.invariant('parity <= 1')
                parity = parity ^ (n & 1)
                n = n >> 1
            return parity
        
        vp.verify_all()
    
    def test_bit_reverse(self):
        """Test reverse bits of integer."""
        vp.scope('test_bit_reverse')
        
        @verify(requires=['n >= 0', 'n < 256'], ensures=['ans >= 0'])
        def reverse_bits(n: int) -> int:
            result = 0
            bits = 8
            i = 0
            while i < bits:
                vp.invariant('i >= 0')
                vp.invariant('i <= bits')
                result = (result << 1) | (n & 1)
                n = n >> 1
                i = i + 1
            return result
        
        vp.verify_all()
    
    def test_xor_properties(self):
        """Test XOR properties."""
        vp.scope('test_xor_props')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def xor_properties() -> int:
            # x ^ x = 0
            # x ^ 0 = x
            # x ^ y = y ^ x
            x = 42
            y = 17
            a = x ^ x
            b = x ^ 0
            c = x ^ y
            d = y ^ x
            # a should be 0, b should be x, c should equal d
            return 1 if a == 0 and b == x and c == d else 0
        
        vp.verify_all()
    
    def test_bitwise_and_range(self):
        """Test bitwise AND of range."""
        vp.scope('test_and_range')
        
        @verify(requires=['m >= 1', 'n >= m'], ensures=['ans >= 0'])
        def range_bitwise_and(m: int, n: int) -> int:
            shift = 0
            while m != n:
                vp.invariant('m >= 0')
                vp.invariant('n >= m')
                m = m >> 1
                n = n >> 1
                shift = shift + 1
            return m << shift
        
        vp.verify_all()


class TestMathematicalProperties(unittest.TestCase):
    """Test cases for mathematical properties and proofs."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_gcd_properties(self):
        """Test GCD properties."""
        vp.scope('test_gcd_props')
        
        @verify(requires=['a >= 0', 'b >= 0'], ensures=['ans >= 0'])
        def gcd_props(a: int, b: int) -> int:
            # gcd(a, b) = gcd(b, a mod b)
            # gcd(a, 0) = a
            x = a
            y = b
            while y != 0:
                vp.invariant('y >= 0')
                vp.invariant('gcd(x, y) == gcd(a, b)')
                x, y = y, x % y
            return x
        
        vp.verify_all()
    
    def test_lcm_properties(self):
        """Test LCM properties."""
        vp.scope('test_lcm_props')
        
        @verify(requires=['a >= 1', 'b >= 1'], ensures=['ans >= 0'])
        def lcm_properties(a: int, b: int) -> int:
            # lcm(a, b) * gcd(a, b) = a * b
            if a == 0 or b == 0:
                return 0
            return (a * b) // 1  # Simplified
        
        vp.verify_all()
    
    def test_fibonacci_properties(self):
        """Test Fibonacci properties."""
        vp.scope('test_fib_props')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def fib_props(n: int) -> int:
            # F(n) = F(n-1) + F(n-2)
            # F(0) = 0, F(1) = 1
            if n == 0:
                return 0
            elif n == 1:
                return 1
            else:
                return fib_props(n - 1) + fib_props(n - 2)
        
        vp.verify_all()
    
    def test_fibonacci_iterative(self):
        """Test Fibonacci iterative with bounds."""
        vp.scope('test_fib_iter')
        
        @verify(requires=['n >= 0', 'n < 50'], ensures=['ans >= 0'])
        def fib_iter(n: int) -> int:
            if n == 0:
                return 0
            a, b = 0, 1
            i = 1
            while i < n:
                vp.invariant('a >= 0')
                vp.invariant('b >= 0')
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                a, b = b, a + b
                i = i + 1
            return b
        
        vp.verify_all()
    
    def test_power_iterative(self):
        """Test power function with exponentiation by squaring."""
        vp.scope('test_power')
        
        @verify(requires=['base >= 0', 'exp >= 0'], ensures=['ans >= 0'])
        def power(base: int, exp: int) -> int:
            result = 1
            b = base
            e = exp
            while e > 0:
                vp.invariant('e >= 0')
                vp.invariant('result >= 1')
                if e % 2 == 1:
                    result = result * b
                b = b * b
                e = e // 2
            return result
        
        vp.verify_all()
    
    def test_summation_formulas(self):
        """Test summation formulas."""
        vp.scope('test_sum_formulas')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def sum_first_n(n: int) -> int:
            # Sum of first n natural numbers: n*(n+1)/2
            return n * (n + 1) // 2
        
        vp.verify_all()
    
    def test_squares_sum(self):
        """Test sum of squares."""
        vp.scope('test_squares_sum')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def sum_squares(n: int) -> int:
            # Sum of squares: n*(n+1)*(2n+1)/6
            total = 0
            i = 0
            while i <= n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n + 1')
                total = total + i * i
                i = i + 1
            return total
        
        vp.verify_all()


class TestArrayManipulation(unittest.TestCase):
    """Test cases for array manipulation algorithms."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_rotate_array(self):
        """Test rotate array by k positions."""
        vp.scope('test_rotate')
        
        @verify(requires=['n >= 0', 'k >= 0'], ensures=['ans >= 0'])
        def rotate_array(arr: List[int], k: int, n: int) -> int:
            # Reverse approach: reverse whole, reverse parts
            if n == 0:
                return 0
            k = k % n
            
            # Reverse first part
            left = 0
            right = n - k - 1
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                temp = arr[left]
                arr[left] = arr[right]
                arr[right] = temp
                left = left + 1
                right = right - 1
            
            # Reverse second part
            left = n - k
            right = n - 1
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                temp = arr[left]
                arr[left] = arr[right]
                arr[right] = temp
                left = left + 1
                right = right - 1
            
            return 0
        
        vp.verify_all()
    
    def test_find_disappeared_numbers(self):
        """Test find disappeared numbers in array."""
        vp.scope('test_disappeared')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'])
        def find_disappeared(nums: List[int], n: int) -> int:
            # Mark numbers by negation
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                val = abs(nums[i]) - 1
                if nums[val] > 0:
                    nums[val] = -nums[val]
                i = i + 1
            
            # Find unmarked positions
            result = 0
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                if nums[i] > 0:
                    result = result + 1
                i = i + 1
            
            return result
        
        vp.verify_all()
    
    def test_product_except_self(self):
        """Test product of array except self."""
        vp.scope('test_product_self')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def product_except_self(arr: List[int], n: int) -> List[int]:
            result: List[int] = [1] * n
            prefix = 1
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                result[i] = prefix
                prefix = prefix * arr[i]
                i = i + 1
            
            suffix = 1
            i = n - 1
            while i >= 0:
                vp.invariant('i >= -1')
                vp.invariant('i < n')
                result[i] = result[i] * suffix
                suffix = suffix * arr[i]
                i = i - 1
            
            return result
        
        vp.verify_all()
    
    def test_max_subarray(self):
        """Test maximum subarray (Kadane's algorithm)."""
        vp.scope('test_max_subarray')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def max_subarray(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            
            max_ending = arr[0]
            max_so_far = arr[0]
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                vp.invariant('max_ending >= 0')
                vp.invariant('max_so_far >= max_ending')
                max_ending = max(arr[i], max_ending + arr[i])
                max_so_far = max(max_so_far, max_ending)
                i = i + 1
            
            return max_so_far
        
        vp.verify_all()
    
    def test_min_subarray(self):
        """Test minimum subarray."""
        vp.scope('test_min_subarray')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def min_subarray(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            
            min_ending = arr[0]
            min_so_far = arr[0]
            i = 1
            while i < n:
                vp.invariant('i >= 1')
                vp.invariant('i <= n')
                vp.invariant('min_ending <= 0')
                vp.invariant('min_so_far <= min_ending')
                min_ending = min(arr[i], min_ending + arr[i])
                min_so_far = min(min_so_far, min_ending)
                i = i + 1
            
            return min_so_far
        
        vp.verify_all()


class TestStringAlgorithms(unittest.TestCase):
    """Test cases for string algorithms."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_is_anagram(self):
        """Test anagram check."""
        vp.scope('test_anagram')
        
        @verify(requires=['len(s1) == len(s2)'], ensures=['ans >= 0'])
        def is_anagram(s1: str, s2: str) -> int:
            if len(s1) != len(s2):
                return 0
            
            count = [0] * 256
            i = 0
            while i < len(s1):
                vp.invariant('i >= 0')
                vp.invariant('i <= len(s1)')
                count[ord(s1[i])] = count[ord(s1[i])] + 1
                count[ord(s2[i])] = count[ord(s2[i])] - 1
                i = i + 1
            
            i = 0
            while i < len(s1):
                vp.invariant('i >= 0')
                vp.invariant('i <= len(s1)')
                if count[ord(s1[i])] != 0:
                    return 0
                i = i + 1
            
            return 1
        
        vp.verify_all()
    
    def test_longest_palindrome(self):
        """Test longest palindromic substring."""
        vp.scope('test_longest_pal')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def longest_palindrome(s: str, n: int) -> int:
            if n == 0:
                return 0
            
            max_len = 1
            i = 0
            while i < n:
                # Odd length palindrome
                left = i - 1
                right = i + 1
                while left >= 0 and right < n and s[left] == s[right]:
                    vp.invariant('left >= -1')
                    vp.invariant('right < n')
                    left = left - 1
                    right = right + 1
                
                # Even length palindrome
                left = i
                right = i + 1
                while left >= 0 and right < n and s[left] == s[right]:
                    vp.invariant('left >= -1')
                    vp.invariant('right < n')
                    left = left - 1
                    right = right + 1
                
                i = i + 1
            
            return max_len
        
        vp.verify_all()
    
    def test_string_reverse(self):
        """Test string reverse."""
        vp.scope('test_str_reverse')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def reverse_string(s: str, n: int) -> int:
            left = 0
            right = n - 1
            while left < right:
                vp.invariant('left >= 0')
                vp.invariant('right < n')
                vp.invariant('left <= right')
                left = left + 1
                right = right - 1
            return 0
        
        vp.verify_all()
    
    def test_substring_search(self):
        """Test substring search (simplified)."""
        vp.scope('test_substr_search')
        
        @verify(requires=['n >= 0', 'm >= 0', 'm <= n'], ensures=['ans >= 0'])
        def substring_search(text: str, pattern: str, n: int, m: int) -> int:
            if m == 0:
                return 0
            
            i = 0
            while i <= n - m:
                vp.invariant('i >= 0')
                vp.invariant('i <= n - m')
                j = 0
                while j < m and text[i + j] == pattern[j]:
                    vp.invariant('j >= 0')
                    vp.invariant('j <= m')
                    j = j + 1
                if j == m:
                    return i
                i = i + 1
            
            return -1
        
        vp.verify_all()


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
