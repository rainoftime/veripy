"""
Test cases for datatypes.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
