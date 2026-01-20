"""
Test cases for stringalgorithms.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
