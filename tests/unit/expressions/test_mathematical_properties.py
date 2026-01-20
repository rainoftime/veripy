"""
Test cases for mathematicalproperties.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
