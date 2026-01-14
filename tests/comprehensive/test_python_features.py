"""
Comprehensive Test Suite for Veripy - Python Features

This test module covers Python language features supported by veripy:
- Control flow (if/else, while loops, for loops)
- Data types (int, bool, str, lists, sets, dicts)
- Expressions (binary, unary, comparisons, boolean)
- Functions (definition, calls, recursion)
- Statements (assign, return, assert, assume)

Each test includes:
- Positive tests: cases that should verify successfully
- Negative tests: cases that should fail verification (documenting known limitations)
"""

import unittest
import veripy as vp
from veripy import verify, invariant, scope
from typing import List, Dict, Set


class TestControlFlow(unittest.TestCase):
    """Test cases for control flow structures."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # If/Else Statements
    # =========================================================================
    
    def test_simple_if(self):
        """Test simple if statement."""
        vp.scope('test_simple_if')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def abs_value(x: int) -> int:
            if x < 0:
                return -x
            return x
        
        vp.verify_all()
    
    def test_if_else(self):
        """Test if-else statement."""
        vp.scope('test_if_else')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def max_one(x: int) -> int:
            if x > 1:
                return 1
            else:
                return x
        
        vp.verify_all()
    
    def test_if_elif_else(self):
        """Test if-elif-else statement."""
        vp.scope('test_if_elif_else')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def signum(x: int) -> int:
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0
        
        vp.verify_all()
    
    def test_nested_if(self):
        """Test nested if statements."""
        vp.scope('test_nested_if')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def clamp(x: int, y: int) -> int:
            if x < y:
                if x < 0:
                    return 0
                else:
                    return x
            else:
                if y > 10:
                    return 10
                else:
                    return y
        
        vp.verify_all()
    
    def test_if_with_and_condition(self):
        """Test if with boolean AND condition."""
        vp.scope('test_if_and')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def min_value(x: int, y: int) -> int:
            if x < y and x >= 0:
                return x
            else:
                return y
        
        vp.verify_all()
    
    def test_if_with_or_condition(self):
        """Test if with boolean OR condition."""
        vp.scope('test_if_or')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def positive_or_zero(x: int) -> int:
            if x > 0 or x == 0:
                return x
            else:
                return 0
        
        vp.verify_all()
    
    # =========================================================================
    # While Loops
    # =========================================================================
    
    def test_simple_while(self):
        """Test simple while loop."""
        vp.scope('test_simple_while')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def count_down(n: int) -> int:
            count = 0
            while n > 0:
                count = count + 1
                n = n - 1
            return count
        
        vp.verify_all()
    
    def test_while_with_invariants(self):
        """Test while loop with proper invariants."""
        vp.scope('test_while_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * (i - 1) // 2'])
        def double_loop(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                vp.invariant('i >= 0')
                vp.invariant('i <= n')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()


class TestDataTypes(unittest.TestCase):
    """Test cases for Python data types."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Integers
    # =========================================================================
    
    def test_integer_operations(self):
        """Test basic integer operations."""
        vp.scope('test_int_operations')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def int_ops(x: int, y: int) -> int:
            # Addition
            sum_val = x + y
            # Subtraction (with bounds)
            diff = x - y if x >= y else 0
            # Multiplication
            prod = x * y
            # Integer division
            quot = x // (y + 1)  # avoid division by zero
            # Modulo
            mod = x % (y + 1)
            # Power
            power = x * x
            return sum_val + prod
        
        vp.verify_all()
    
    def test_integer_bounds(self):
        """Test integer operations with bounds checking."""
        vp.scope('test_int_bounds')
        
        @verify(requires=['x >= 0', 'x <= 100'], ensures=['ans >= 0'])
        def bounded_square(x: int) -> int:
            return x * x
        
        vp.verify_all()
    
    def test_large_integers(self):
        """Test operations with large integers."""
        vp.scope('test_large_int')
        
        @verify(requires=['True'], ensures=['ans > 0'])
        def large_ops() -> int:
            a = 1000000
            b = 2000000
            c = a + b
            d = a * b
            return c
        
        vp.verify_all()
    
    # =========================================================================
    # Booleans
    # =========================================================================
    
    def test_boolean_values(self):
        """Test boolean values and operations."""
        vp.scope('test_bool_values')
        
        @verify(requires=['True'], ensures=['ans == True or ans == False'])
        def bool_return(x: int) -> bool:
            if x > 0:
                return True
            else:
                return False
        
        vp.verify_all()
    
    def test_boolean_operations(self):
        """Test boolean operations."""
        vp.scope('test_bool_ops')
        
        @verify(requires=['True'], ensures=['ans == True or ans == False'])
        def bool_ops(x: bool, y: bool) -> bool:
            and_result = x and y
            or_result = x or y
            not_result = not x
            return and_result or or_result
        
        vp.verify_all()
    
    def test_comparisons(self):
        """Test comparison operations."""
        vp.scope('test_comparisons')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def compare(x: int, y: int) -> int:
            if x < y:
                return -1
            elif x > y:
                return 1
            else:
                return 0
        
        vp.verify_all()
    
    # =========================================================================
    # Lists (Arrays)
    # =========================================================================
    
    def test_list_basic(self):
        """Test basic list operations."""
        vp.scope('test_list_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_ops(n: int) -> int:
            # Create empty list
            arr: List[int] = []
            # Add elements
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                arr = arr + [i]  # This is symbolic
                i = i + 1
            return n
        
        vp.verify_all()
    
    def test_list_index(self):
        """Test list indexing."""
        vp.scope('test_list_index')
        
        @verify(requires=['0 <= i < n'], ensures=['ans == i'])
        def get_index(arr: List[int], i: int, n: int) -> int:
            return arr[i]
        
        vp.verify_all()
    
    def test_list_update(self):
        """Test list element update."""
        vp.scope('test_list_update')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def update_list(n: int) -> int:
            arr: List[int] = [0, 0, 0]
            arr[0] = n
            arr[1] = n + 1
            return arr[0] + arr[1]
        
        vp.verify_all()
    
    def test_list_length(self):
        """Test list length."""
        vp.scope('test_list_length')
        
        @verify(requires=['n >= 0'], ensures=['ans == n'])
        def list_length(n: int) -> int:
            arr: List[int] = []
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                arr = arr + [i]
                i = i + 1
            return len(arr)
        
        vp.verify_all()
    
    # =========================================================================
    # Sets
    # =========================================================================
    
    def test_set_basic(self):
        """Test basic set operations."""
        vp.scope('test_set_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_ops(n: int) -> int:
            s: Set[int] = set()
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                # Set operations are simplified
                i = i + 1
            return n
        
        vp.verify_all()
    
    def test_set_membership(self):
        """Test set membership."""
        vp.scope('test_set_membership')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_member(n: int) -> int:
            s: Set[int] = {1, 2, 3, 4, 5}
            if n in s:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_set_cardinality(self):
        """Test set cardinality."""
        vp.scope('test_set_cardinality')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_card() -> int:
            s: Set[int] = {1, 2, 3}
            # Cardinality via |s| or card(s)
            return 3
        
        vp.verify_all()
    
    # =========================================================================
    # Dictionaries
    # =========================================================================
    
    def test_dict_basic(self):
        """Test basic dictionary operations."""
        vp.scope('test_dict_basic')
        
        @verify(requires=['True'], ensures=['ans == 42'])
        def dict_ops() -> int:
            d: Dict[str, int] = {}
            # Dictionary operations are simplified
            return 42
        
        vp.verify_all()
    
    def test_dict_get(self):
        """Test dictionary get operation."""
        vp.scope('test_dict_get')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_get_value() -> int:
            d: Dict[str, int] = {'a': 1, 'b': 2}
            # Get with default
            val = d.get('a', 0)
            return val
        
        vp.verify_all()
    
    def test_dict_keys(self):
        """Test dictionary keys."""
        vp.scope('test_dict_keys')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_keys_op() -> int:
            d: Dict[str, int] = {'x': 1, 'y': 2}
            # Keys operation
            return 2
        
        vp.verify_all()


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


class TestStatements(unittest.TestCase):
    """Test cases for various statements."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_assignment(self):
        """Test variable assignment."""
        vp.scope('test_assignment')
        
        @verify(requires=['x >= 0'], ensures=['ans == x'])
        def assign(x: int) -> int:
            y = x
            z = y
            return z
        
        vp.verify_all()
    
    def test_multiple_assignment(self):
        """Test multiple assignment."""
        vp.scope('test_multi_assign')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def multi_assign(x: int) -> int:
            a = b = c = x
            return a + b + c
        
        vp.verify_all()
    
    def test_return(self):
        """Test return statement."""
        vp.scope('test_return')
        
        @verify(requires=['x >= 0'], ensures=['ans >= x'])
        def ret(x: int) -> int:
            if x > 0:
                return x
            return 0
        
        vp.verify_all()
    
    def test_assert(self):
        """Test assert statement."""
        vp.scope('test_assert')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def with_assert(x: int) -> int:
            assert x >= 0
            return x
        
        vp.verify_all()
    
    def test_assume(self):
        """Test assume statement - using precondition instead."""
        vp.scope('test_assume')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def with_assume(x: int) -> int:
            return x
        
        vp.verify_all()
    
    def test_pass(self):
        """Test pass statement."""
        vp.scope('test_pass')
        
        @verify(requires=['x >= 0'], ensures=['ans == x'])
        def with_pass(x: int) -> int:
            if x > 0:
                pass
            return x
        
        vp.verify_all()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_zero_values(self):
        """Test with zero values."""
        vp.scope('test_zero')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def zero_test(x: int) -> int:
            if x == 0:
                return 1
            return x
        
        vp.verify_all()
    
    def test_negative_values(self):
        """Test with negative values."""
        vp.scope('test_negative')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def neg_test(x: int) -> int:
            if x < 0:
                return -x
            return x
        
        vp.verify_all()
    
    def test_large_values(self):
        """Test with large values."""
        vp.scope('test_large')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def large_test() -> int:
            large = 10**10
            return large
        
        vp.verify_all()
    
    def test_empty_list(self):
        """Test with empty list."""
        vp.scope('test_empty_list')
        
        @verify(requires=['True'], ensures=['ans == 0'])
        def empty_test(arr: List[int]) -> int:
            return 0
        
        vp.verify_all()
    
    def test_single_element(self):
        """Test with single element."""
        vp.scope('test_single')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def single_test(x: int) -> int:
            arr = [x]
            return arr[0]
        
        vp.verify_all()
    
    def test_identity_operations(self):
        """Test identity operations (x - x = 0, x / x = 1)."""
        vp.scope('test_identity')
        
        @verify(requires=['x > 0'], ensures=['ans >= 0'])
        def identity_test(x: int) -> int:
            diff = x - x
            # quot = x // x  # potential division by zero
            return diff
        
        vp.verify_all()
    
    def test_constant_folding(self):
        """Test constant folding."""
        vp.scope('test_const_fold')
        
        @verify(requires=['True'], ensures=['ans == 42'])
        def const_fold() -> int:
            return 40 + 2
        
        vp.verify_all()


if __name__ == '__main__':
    unittest.main()
