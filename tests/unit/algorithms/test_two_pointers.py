"""
Test cases for twopointers.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
