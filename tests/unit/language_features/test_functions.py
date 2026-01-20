"""
Test cases for functions.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestFunctions(unittest.TestCase):
    """Test cases for function definitions and calls."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Basic Function Calls
    # =========================================================================
    
    def test_single_function(self):
        """Test single function definition and call."""
        vp.scope('test_single_function')
        
        @verify(requires=['x >= 0'], ensures=['ans >= x'])
        def inc(x: int) -> int:
            return x + 1
        
        vp.verify_all()
    
    def test_function_call(self):
        """Test function call within another function."""
        vp.scope('test_function_call')
        
        @verify(requires=['x >= 0'], ensures=['ans >= x'])
        def inc(x: int) -> int:
            return x + 1
        
        @verify(requires=['x >= 0'], ensures=['ans >= x'])
        def double_inc(x: int) -> int:
            y = inc(x)
            return inc(y)
        
        vp.verify_all()
    
    def test_multiple_calls(self):
        """Test multiple function calls."""
        vp.scope('test_multiple_calls')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def square(x: int) -> int:
            return x * x
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def fourth_power(x: int) -> int:
            s1 = square(x)
            s2 = square(x)
            return s1 + s2
        
        vp.verify_all()
    
    def test_call_with_expression(self):
        """Test function calls with expressions as arguments."""
        vp.scope('test_call_expr')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def add(x: int, y: int) -> int:
            return x + y
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def complex_call(x: int, y: int) -> int:
            return add(x + 1, y * 2)
        
        vp.verify_all()
    
    # =========================================================================
    # Recursive Functions
    # =========================================================================
    
    def test_simple_recursion(self):
        """Test simple recursive function."""
        vp.scope('test_simple_recursion')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def factorial(n: int) -> int:
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)
        
        vp.verify_all()
    
    def test_recursion_with_call(self):
        """Test recursion with additional function call."""
        vp.scope('test_recursion_call')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def factorial(n: int) -> int:
            if n == 0:
                return 1
            else:
                return n * factorial(n - 1)
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def factorial_plus_one(n: int) -> int:
            f = factorial(n)
            return f + 1
        
        vp.verify_all()
    
    def test_double_recursion(self):
        """Test mutually recursive functions."""
        vp.scope('test_double_recursion')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def fib(n: int) -> int:
            if n == 0:
                return 0
            elif n == 1:
                return 1
            else:
                return fib(n - 1) + fib(n - 2)
        
        vp.verify_all()
    
    def test_nested_recursion(self):
        """Test nested recursion."""
        vp.scope('test_nested_recursion')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans >= 0'], decreases='n')
        def ackermann(m: int, n: int) -> int:
            if m == 0:
                return n + 1
            elif n == 0:
                return ackermann(m - 1, 1)
            else:
                return ackermann(m - 1, ackermann(m, n - 1))
        
        vp.verify_all()
    
    def test_recursion_with_loop(self):
        """Test recursion combined with loop."""
        vp.scope('test_recursion_loop')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def sum_to_n(n: int) -> int:
            if n == 0:
                return 0
            else:
                result = 0
                i = 0
                while i < n:
                    vp.invariant('i >= 0')
                    vp.invariant('i <= n')
                    result = result + i
                    i = i + 1
                return n + result
        
        vp.verify_all()
    
    # =========================================================================
    # Type Annotations
    # =========================================================================
    
    def test_type_annotations(self):
        """Test functions with type annotations."""
        vp.scope('test_type_annotations')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def typed_add(x: int, y: int) -> int:
            return x + y
        
        vp.verify_all()
    
    def test_complex_type_annotations(self):
        """Test functions with complex type annotations."""
        vp.scope('test_complex_types')
        
        @verify(requires=['len(xs) > 0'], ensures=['ans >= 0'])
        def sum_list(xs: List[int]) -> int:
            total = 0
            i = 0
            while i < len(xs):
                vp.invariant('i >= 0')
                vp.invariant('i <= len(xs)')
                total = total + xs[i]
                i = i + 1
            return total
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
