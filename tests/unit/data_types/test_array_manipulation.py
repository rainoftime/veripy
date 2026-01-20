"""
Test cases for arraymanipulation.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
