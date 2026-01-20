"""
Test cases for bitmanipulation.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
