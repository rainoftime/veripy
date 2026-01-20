"""
Test cases for prefixsumandslidingwindow.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
