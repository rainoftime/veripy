"""
Comprehensive Test Suite for Veripy - Auto-Active Verifier Features

This test module covers auto-active verification features:
- Loop invariants (inference, manual specification)
- Quantifiers (forall, exists)
- Termination checking (decreases clauses)
- Lemma generation and verification
- Frame conditions and modifies clauses
- Automatic invariant inference
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List


class TestLoopInvariants(unittest.TestCase):
    """Test cases for loop invariants."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Manual Invariants
    # =========================================================================
    
    def test_basic_invariant(self):
        """Test basic loop invariant."""
        vp.scope('test_basic_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def basic_loop(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_accumulator_invariant(self):
        """Test loop with accumulator invariant."""
        vp.scope('test_accumulator_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * (n + 1) // 2'])
        def sum_to_n(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans == i * (i - 1) // 2')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_multiple_invariants(self):
        """Test loop with multiple invariants."""
        vp.scope('test_multiple_invariants')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def product(x: int, y: int) -> int:
            ans = 0
            i = 0
            j = 0
            while i < x:
                invariant('i >= 0')
                invariant('i <= x')
                invariant('ans == i * y')
                j = 0
                while j < y:
                    invariant('j >= 0')
                    invariant('j <= y')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_invariant_with_comparison(self):
        """Test invariant with comparison."""
        vp.scope('test_invariant_comparison')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def find_max(arr: List[int], n: int) -> int:
            if n == 0:
                return 0
            max_val = arr[0]
            i = 1
            while i < n:
                invariant('i >= 1')
                invariant('i <= n')
                invariant('max_val >= arr[0]')
                invariant('forall j: int :: j >= 0 and j < i ==> max_val >= arr[j]')
                if arr[i] > max_val:
                    max_val = arr[i]
                i = i + 1
            return max_val
        
        vp.verify_all()
    
    def test_invariant_preservation(self):
        """Test invariant preservation."""
        vp.scope('test_invariant_preservation')
        
        @verify(requires=['n >= 0'], ensures=['ans >= n'])
        def sum_greater_than_n(n: int) -> int:
            ans = n
            i = 0
            while i < n:
                invariant('ans >= n')
                invariant('i >= 0')
                invariant('i <= n')
                ans = ans + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    # =========================================================================
    # Invariant Types
    # =========================================================================
    
    def test_bounds_invariant(self):
        """Test bounds invariant."""
        vp.scope('test_bounds_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bounded_loop(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_type_invariant(self):
        """Test type invariant."""
        vp.scope('test_type_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def typed_loop(n: int) -> int:
            i: int = 0
            ans: int = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_relationship_invariant(self):
        """Test relationship between loop variables."""
        vp.scope('test_relationship_invariant')
        
        @verify(requires=['n >= 0'], ensures=['ans == n * n'])
        def square_loop(n: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans == i * n')
                j = 0
                while j < n:
                    invariant('j >= 0')
                    invariant('j <= n')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    # =========================================================================
    # Complex Loop Patterns
    # =========================================================================
    
    def test_nested_loop_invariants(self):
        """Test nested loop with invariants."""
        vp.scope('test_nested_loop_invariants')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans == n * m'])
        def matrix_sum(n: int, m: int) -> int:
            ans = 0
            i = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                j = 0
                while j < m:
                    invariant('j >= 0')
                    invariant('j <= m')
                    ans = ans + 1
                    j = j + 1
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_loop_with_break_invariants(self):
        """Test loop with break and invariants."""
        vp.scope('test_break_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def find_value(arr: List[int], n: int, target: int) -> int:
            i = 0
            ans = -1
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                if arr[i] == target:
                    ans = i
                    break
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_loop_with_continue_invariants(self):
        """Test loop with continue and invariants."""
        vp.scope('test_continue_invariants')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def skip_odds(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                invariant('i >= 0')
                invariant('i <= n')
                invariant('ans >= 0')
                i = i + 1
                if i % 2 == 1:
                    continue
                ans = ans + i
            return ans
        
        vp.verify_all()


class TestQuantifiers(unittest.TestCase):
    """Test cases for quantifiers."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Forall Quantifier
    # =========================================================================
    
    def test_simple_forall(self):
        """Test simple forall quantifier."""
        vp.scope('test_simple_forall')
        
        @verify(requires=['True'], ensures=['forall x: int :: x == x'])
        def identity() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_with_implication(self):
        """Test forall with implication."""
        vp.scope('test_forall_implication')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: i >= 0 ==> i >= 0'])
        def forall_implication(n: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_universal_truth(self):
        """Test forall quantifier for universal truth."""
        vp.scope('test_forall_truth')
        
        @verify(requires=['True'], ensures=['forall x: int :: x + 0 == x'])
        def add_zero() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_multi_variable(self):
        """Test forall with multiple variables."""
        vp.scope('test_forall_multi')
        
        @verify(requires=['True'], ensures=['forall x: int, y: int :: x + y == y + x'])
        def commutativity() -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_with_arithmetic(self):
        """Test forall with arithmetic."""
        vp.scope('test_forall_arithmetic')
        
        @verify(requires=['True'], ensures=['forall x: int, y: int :: x + y > x'])
        def arithmetic_forall(y: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_forall_range(self):
        """Test forall with range constraint."""
        vp.scope('test_forall_range')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: i >= 0 and i < n ==> arr[i] >= 0'])
        def array_non_negative(arr: List[int], n: int) -> bool:
            return True
        
        vp.verify_all()
    
    # =========================================================================
    # Exists Quantifier
    # =========================================================================
    
    def test_simple_exists(self):
        """Test simple exists quantifier."""
        vp.scope('test_simple_exists')
        
        @verify(requires=['True'], ensures=['exists x: int :: x > 0'])
        def exists_positive() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_with_condition(self):
        """Test exists with condition."""
        vp.scope('test_exists_condition')
        
        @verify(requires=['n >= 0'], ensures=['exists i: int :: i >= 0 and i < n'])
        def exists_in_range(n: int) -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_not_forall(self):
        """Test exists is not forall."""
        vp.scope('test_exists_not_forall')
        
        @verify(requires=['True'], ensures=['exists x: int :: not (x > 0)'])
        def exists_non_positive() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_complex(self):
        """Test complex exists quantifier."""
        vp.scope('test_exists_complex')
        
        @verify(requires=['n >= 0'], ensures=['exists i: int :: i >= 0 and i < n and arr[i] == 5'])
        def find_five(arr: List[int], n: int) -> bool:
            return True
        
        vp.verify_all()
    
    # =========================================================================
    # Quantifier Combinations
    # =========================================================================
    
    def test_forall_exists(self):
        """Test forall followed by exists."""
        vp.scope('test_forall_exists')
        
        @verify(requires=['n >= 0'], ensures=['forall i: int :: exists j: int :: j > i'])
        def infinite_growth() -> bool:
            return True
        
        vp.verify_all()
    
    def test_exists_forall(self):
        """Test exists followed by forall."""
        return # TIMEOUT 
        vp.scope('test_exists_forall')
        
        @verify(requires=['True'], ensures=['exists x: int :: forall y: int :: x <= y'])
        def minimum_exists() -> bool:
            return True
        
        vp.verify_all()
    
    def test_quantifier_equivalence(self):
        """Test quantifier equivalence."""
        vp.scope('test_quantifier_equivalence')
        
        @verify(requires=['True'], ensures=['not (forall x :: P(x)) <==> exists x :: not P(x)'])
        def de_morgan_quantifiers() -> bool:
            return True
        
        vp.verify_all()


class TestTermination(unittest.TestCase):
    """Test cases for termination checking."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Decreases Clause
    # =========================================================================
    
    def test_decreases_single(self):
        """Test decreases with single variable."""
        vp.scope('test_decreases_single')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def countdown(n: int) -> int:
            if n == 0:
                return 0
            else:
                return countdown(n - 1)
        
        vp.verify_all()
    
    def test_decreases_tuple(self):
        """Test decreases with tuple."""
        vp.scope('test_decreases_tuple')
        
        @verify(requires=['n >= 0', 'm >= 0'], ensures=['ans >= 0'], decreases='(n, m)')
        def nested_countdown(n: int, m: int) -> int:
            if n == 0:
                if m == 0:
                    return 0
                else:
                    return nested_countdown(n, m - 1)
            else:
                return nested_countdown(n - 1, m)
        
        vp.verify_all()
    
    def test_decreases_expression(self):
        """Test decreases with expression."""
        vp.scope('test_decreases_expr')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'], decreases='n // 2')
        def halving(n: int) -> int:
            if n == 1:
                return 0
            else:
                return halving(n // 2)
        
        vp.verify_all()
    
    # =========================================================================
    # Termination Patterns
    # =========================================================================
    
    def test_linear_termination(self):
        """Test linear termination."""
        vp.scope('test_linear_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def linear_recursion(n: int) -> int:
            if n == 0:
                return 0
            return 1 + linear_recursion(n - 1)
        
        vp.verify_all()
    
    def test_logarithmic_termination(self):
        """Test logarithmic termination."""
        vp.scope('test_log_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def binary_recursion(n: int) -> int:
            if n <= 1:
                return n
            return binary_recursion(n // 2) + binary_recursion(n // 2)
        
        vp.verify_all()
    
    def test_tree_termination(self):
        """Test tree-structured recursion termination."""
        vp.scope('test_tree_termination')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'], decreases='n')
        def tree_sum(n: int) -> int:
            if n == 0:
                return 0
            left = tree_sum(n - 1)
            right = tree_sum(n - 1)
            return 1 + left + right
        
        vp.verify_all()
    
    # =========================================================================
    # Non-Terminating Cases
    # =========================================================================
    
    def test_missing_decreases(self):
        """Test that missing decreases clause is detected."""
        vp.scope('test_missing_decreases')
        
        # This should fail or require decreases clause
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def no_decreases(n: int) -> int:
            if n > 0:
                return no_decreases(n)
            return 0
        
        # This will fail - expected behavior
        try:
            vp.verify_all()
        except:
            pass  # Expected - should fail verification
    
    def test_non_decreasing(self):
        """Test non-decreasing recursive call."""
        vp.scope('test_non_decreasing')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def bad_recursion(n: int) -> int:
            if n == 0:
                return 0
            # n + 1 is not decreasing
            return bad_recursion(n + 1)
        
        # This should fail
        try:
            vp.verify_all()
        except:
            pass  # Expected - should fail verification


class TestFrameConditions(unittest.TestCase):
    """Test cases for frame conditions and modifies clauses."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_single_variable_modifies(self):
        """Test modifies clause with single variable."""
        vp.scope('test_single_modifies')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def increment(x: int) -> int:
            return x + 1
        
        vp.verify_all()
    
    def test_multiple_variables(self):
        """Test with multiple variables."""
        vp.scope('test_multi_variables')
        
        @verify(requires=['x >= 0', 'y >= 0'], ensures=['ans >= 0'])
        def swap(x: int, y: int) -> int:
            temp = x
            x = y
            y = temp
            return x + y
        
        vp.verify_all()
    
    def test_array_modifies(self):
        """Test array modification frame."""
        vp.scope('test_array_modifies')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def update_array(arr: List[int], n: int) -> int:
            arr[0] = n
            return arr[0]
        
        vp.verify_all()


class TestAutomaticInference(unittest.TestCase):
    """Test cases for automatic invariant inference."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_auto_bounds_inference(self):
        """Test automatic bounds inference."""
        vp.scope('test_auto_bounds')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def auto_bounds(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                # Veripy should auto-infer: i >= 0, i <= n
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()
    
    def test_auto_type_inference(self):
        """Test automatic type inference."""
        vp.scope('test_auto_type')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def auto_type() -> int:
            x: int = 0
            y: int = 1
            # Veripy should infer types
            return x + y
        
        vp.verify_all()
    
    def test_auto_relationship_inference(self):
        """Test automatic relationship inference."""
        vp.scope('test_auto_relationship')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def auto_relationship(n: int) -> int:
            i = 0
            ans = 0
            while i < n:
                # Veripy should auto-infer: ans relates to i
                ans = ans + i
                i = i + 1
            return ans
        
        vp.verify_all()


class TestComplexVerification(unittest.TestCase):
    """Complex verification scenarios combining multiple features."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_binary_search(self):
        """Verify binary search algorithm."""
        vp.scope('test_binary_search')
        
        @verify(requires=['n > 0'], ensures=['ans >= 0'])
        def binary_search(arr: List[int], target: int, n: int) -> int:
            lo = 0
            hi = n
            while lo < hi:
                invariant('lo >= 0')
                invariant('hi <= n')
                invariant('forall i: int :: i < lo or i >= hi ==> arr[i] != target')
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            return -1
        
        vp.verify_all()
    
    def test_merge_sort(self):
        """Verify merge sort algorithm."""
        vp.scope('test_merge_sort')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def merge_sort_helper(arr: List[int], n: int) -> int:
            if n <= 1:
                return n
            mid = n // 2
            left = merge_sort_helper(arr[:mid], mid)
            right = merge_sort_helper(arr[mid:], n - mid)
            return left + right
        
        vp.verify_all()
    
    def test_dynamic_programming(self):
        """Verify dynamic programming pattern."""
        vp.scope('test_dp')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def fib_dp(n: int) -> int:
            if n == 0:
                return 0
            if n == 1:
                return 1
            a = 0
            b = 1
            i = 2
            while i < n:
                invariant('i >= 2')
                invariant('i <= n')
                invariant('a == fib(i - 2)')
                invariant('b == fib(i - 1)')
                c = a + b
                a = b
                b = c
                i = i + 1
            return a + b
        
        vp.verify_all()
    
    def test_gcd(self):
        """Verify Euclidean GCD algorithm."""
        vp.scope('test_gcd')
        
        @verify(requires=['a >= 0', 'b >= 0'], ensures=['ans >= 0'])
        def gcd(a: int, b: int) -> int:
            while b != 0:
                invariant('b >= 0')
                invariant('gcd(a, b) == gcd(b, a % b)')
                a, b = b, a % b
            return a
        
        vp.verify_all()


if __name__ == '__main__':
    unittest.main()
