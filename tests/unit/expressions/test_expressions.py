"""
Test cases for expressions.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


class TestExpressions(unittest.TestCase):
    """Test cases for Python expressions."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Binary Operations
    # =========================================================================
    
    def test_arithmetic_ops(self):
        """Test arithmetic operations."""
        vp.scope('test_arith_ops')
        
        @verify(requires=['x >= 0', 'y > 0'], ensures=['ans >= 0'])
        def arithmetic(x: int, y: int) -> int:
            a = x + y
            b = x - y
            c = x * y
            d = x // y
            e = x % y
            f = x ** 2
            return a + b + c + d + e + f
        
        vp.verify_all()
    
    def test_division_properties(self):
        """Test division with properties."""
        vp.scope('test_division_props')
        
        @verify(requires=['y > 0'], ensures=['ans >= 0'])
        def div_props(x: int, y: int) -> int:
            q = x // y
            r = x % y
            # x = q * y + r, 0 <= r < y
            return q
        
        vp.verify_all()
    
    def test_power_properties(self):
        """Test power operation properties."""
        vp.scope('test_power_props')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 1'])
        def power_props(x: int, n: int) -> int:
            # x^0 = 1
            # x^(m+n) = x^m * x^n
            result = 1
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                vp.invariant('result >= 1')
                result = result * x
                i = i + 1
            return result
        
        vp.verify_all()
    
    # =========================================================================
    # Unary Operations
    # =========================================================================
    
    def test_unary_minus(self):
        """Test unary minus operation."""
        vp.scope('test_unary_minus')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def unary_ops(x: int) -> int:
            neg = -x
            pos = +x
            double_neg = -neg
            return double_neg
        
        vp.verify_all()
    
    def test_unary_not(self):
        """Test unary not operation."""
        vp.scope('test_unary_not')
        
        @verify(requires=['True'], ensures=['ans == True or ans == False'])
        def not_ops(x: bool) -> bool:
            return not x
        
        vp.verify_all()
    
    # =========================================================================
    # Comparison Operations
    # =========================================================================
    
    def test_all_comparisons(self):
        """Test all comparison operations."""
        vp.scope('test_all_comparisons')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def comparisons(x: int, y: int) -> int:
            results = 0
            if x < y:
                results = results + 1
            if x <= y:
                results = results + 1
            if x > y:
                results = results + 1
            if x >= y:
                results = results + 1
            if x == y:
                results = results + 1
            if x != y:
                results = results + 1
            return results
        
        vp.verify_all()
    
    def test_chained_comparisons(self):
        """Test chained comparisons."""
        vp.scope('test_chained_comparisons')
        
        @verify(requires=['x >= 0', 'x <= 10'], ensures=['ans >= 0'])
        def chained(x: int) -> int:
            if 0 <= x <= 10:
                return 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Boolean Operations
    # =========================================================================
    
    def test_bool_and(self):
        """Test boolean AND."""
        vp.scope('test_bool_and')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def and_op(x: int) -> int:
            if x >= 0 and x <= 10:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_bool_or(self):
        """Test boolean OR."""
        vp.scope('test_bool_or')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def or_op(x: int) -> int:
            if x < 0 or x > 10:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_bool_not(self):
        """Test boolean NOT."""
        vp.scope('test_bool_not')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def not_op(x: int) -> int:
            if not (x < 0):
                return 1
            return 0
        
        vp.verify_all()
    
    def test_de_morgans_law(self):
        """Test De Morgan's laws."""
        vp.scope('test_de_morgan')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def de_morgan(x: int, y: int) -> int:
            # not (A and B) == (not A) or (not B)
            a = x >= 0
            b = y >= 0
            left = not (a and b)
            right = (not a) or (not b)
            if left == right:
                return 1
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Membership Tests
    # =========================================================================
    
    def test_membership_in_list(self):
        """Test membership in list."""
        vp.scope('test_membership_list')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def in_list(n: int) -> int:
            arr = [1, 2, 3, 4, 5]
            if n in arr:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_membership_not_in(self):
        """Test membership with not in."""
        vp.scope('test_not_membership')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def not_in_list(n: int) -> int:
            arr = [1, 2, 3]
            if n not in arr:
                return 1
            return 0
        
        vp.verify_all()


if __name__ == "__main__":
    unittest.main()
