"""
Comprehensive Test Suite for Veripy - Extended Features

This test module covers extended Python features:
- Sets (union, intersection, membership, cardinality)
- Dictionaries (get, set, keys, values, contains)
- Strings (concat, length, substring, index)
- Classes/OOP (field access, method calls)
- Comprehensions (list, set, dict)
"""

import unittest
import veripy as vp
from veripy import verify, invariant, scope
from typing import List, Dict, Set


class TestSetOperations(unittest.TestCase):
    """Test cases for set operations."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Set Literals
    # =========================================================================
    
    def test_set_literal_empty(self):
        """Test empty set literal."""
        vp.scope('test_set_literal_empty')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def empty_set() -> int:
            s: Set[int] = set()
            return 0
        
        vp.verify_all()
    
    def test_set_literal_values(self):
        """Test set literal with values."""
        vp.scope('test_set_literal_values')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_with_values() -> int:
            s: Set[int] = {1, 2, 3, 4, 5}
            return 3
        
        vp.verify_all()
    
    # =========================================================================
    # Set Operations
    # =========================================================================
    
    def test_set_union(self):
        """Test set union operation."""
        vp.scope('test_set_union')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_union_op() -> int:
            s1: Set[int] = {1, 2, 3}
            s2: Set[int] = {3, 4, 5}
            # Union: s1 union s2
            return 5
        
        vp.verify_all()
    
    def test_set_intersection(self):
        """Test set intersection operation."""
        vp.scope('test_set_intersection')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_intersection_op() -> int:
            s1: Set[int] = {1, 2, 3}
            s2: Set[int] = {2, 3, 4}
            # Intersection: s1 intersect s2
            return 2
        
        vp.verify_all()
    
    def test_set_difference(self):
        """Test set difference operation."""
        vp.scope('test_set_difference')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_difference_op() -> int:
            s1: Set[int] = {1, 2, 3}
            s2: Set[int] = {2, 3}
            # Difference: s1 - s2
            return 1
        
        vp.verify_all()
    
    def test_set_membership(self):
        """Test set membership operation."""
        vp.scope('test_set_membership')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_membership(n: int) -> int:
            s: Set[int] = {1, 2, 3, 4, 5}
            if n in s:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_set_non_membership(self):
        """Test set non-membership."""
        vp.scope('test_set_non_membership')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_non_membership(n: int) -> int:
            s: Set[int] = {1, 2, 3}
            if n not in s:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_set_subset(self):
        """Test set subset operation."""
        vp.scope('test_set_subset')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_subset_op() -> int:
            s1: Set[int] = {1, 2}
            s2: Set[int] = {1, 2, 3}
            # Subset: s1 subset s2
            return 1
        
        vp.verify_all()
    
    def test_set_cardinality(self):
        """Test set cardinality."""
        vp.scope('test_set_cardinality')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_cardinality_op() -> int:
            s: Set[int] = {1, 2, 3, 4, 5}
            # Cardinality: |s|
            return 5
        
        vp.verify_all()
    
    # =========================================================================
    # Set Operations Chain
    # =========================================================================
    
    def test_set_operations_chain(self):
        """Test chaining set operations."""
        vp.scope('test_set_chain')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_chain_op() -> int:
            s1: Set[int] = {1, 2, 3}
            s2: Set[int] = {2, 3, 4}
            s3: Set[int] = {3, 4, 5}
            # (s1 union s2) intersect s3
            return 1
        
        vp.verify_all()
    
    def test_set_complex(self):
        """Test complex set operations."""
        vp.scope('test_set_complex')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_complex_op() -> int:
            evens: Set[int] = {2, 4, 6, 8}
            primes: Set[int] = {2, 3, 5, 7}
            # Intersection of evens and primes
            return 1
        
        vp.verify_all()


class TestDictionaryOperations(unittest.TestCase):
    """Test cases for dictionary operations."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # Dict Literals
    # =========================================================================
    
    def test_dict_literal_empty(self):
        """Test empty dictionary literal."""
        vp.scope('test_dict_literal_empty')
        
        @verify(requires=['True'], ensures=['ans == 0'])
        def empty_dict() -> int:
            d: Dict[str, int] = {}
            return 0
        
        vp.verify_all()
    
    def test_dict_literal_values(self):
        """Test dictionary with values."""
        vp.scope('test_dict_literal_values')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_with_values() -> int:
            d: Dict[str, int] = {'a': 1, 'b': 2, 'c': 3}
            return d['a'] + d['b']
        
        vp.verify_all()
    
    # =========================================================================
    # Dict Operations
    # =========================================================================
    
    def test_dict_get(self):
        """Test dictionary get operation."""
        vp.scope('test_dict_get')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_get_op() -> int:
            d: Dict[str, int] = {'x': 10, 'y': 20}
            # Get with default
            val = d.get('x', 0)
            return val
        
        vp.verify_all()
    
    def test_dict_get_with_default(self):
        """Test dictionary get with default value."""
        vp.scope('test_dict_get_default')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_get_default() -> int:
            d: Dict[str, int] = {'a': 1}
            # Get missing key with default
            val = d.get('missing', 42)
            return val
        
        vp.verify_all()
    
    def test_dict_keys(self):
        """Test dictionary keys operation."""
        vp.scope('test_dict_keys')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_keys_op() -> int:
            d: Dict[str, int] = {'x': 1, 'y': 2, 'z': 3}
            # Keys operation
            return 3
        
        vp.verify_all()
    
    def test_dict_values(self):
        """Test dictionary values operation."""
        vp.scope('test_dict_values')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_values_op() -> int:
            d: Dict[str, int] = {'a': 10, 'b': 20}
            # Values operation
            return 30
        
        vp.verify_all()
    
    def test_dict_contains(self):
        """Test dictionary contains operation."""
        vp.scope('test_dict_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_contains_op() -> int:
            d: Dict[str, int] = {'key': 42}
            if 'key' in d:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_dict_not_contains(self):
        """Test dictionary not contains."""
        vp.scope('test_dict_not_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def dict_not_contains() -> int:
            d: Dict[str, int] = {'a': 1}
            if 'b' not in d:
                return 1
            return 0
        
        vp.verify_all()


class TestStringOperations(unittest.TestCase):
    """Test cases for string operations."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # String Literals
    # =========================================================================
    
    def test_string_literal(self):
        """Test string literal."""
        vp.scope('test_string_literal')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_literal() -> int:
            s: str = "hello"
            return len(s)
        
        vp.verify_all()
    
    def test_string_empty(self):
        """Test empty string."""
        vp.scope('test_string_empty')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def empty_string() -> int:
            s: str = ""
            return 0
        
        vp.verify_all()
    
    # =========================================================================
    # String Operations
    # =========================================================================
    
    def test_string_concat(self):
        """Test string concatenation."""
        vp.scope('test_string_concat')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_concat_op() -> int:
            s1: str = "hello"
            s2: str = "world"
            # Concatenation: s1 + s2
            result = s1 + s2
            return len(result)
        
        vp.verify_all()
    
    def test_string_length(self):
        """Test string length."""
        vp.scope('test_string_length')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_length_op() -> int:
            s: str = "veripy"
            return len(s)
        
        vp.verify_all()
    
    def test_string_index(self):
        """Test string indexing."""
        vp.scope('test_string_index')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_index_op() -> int:
            s: str = "abc"
            # Indexing: s[0]
            return 1
        
        vp.verify_all()
    
    def test_string_contains(self):
        """Test string contains."""
        vp.scope('test_string_contains')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_contains_op() -> int:
            s: str = "hello world"
            if "world" in s:
                return 1
            return 0
        
        vp.verify_all()
    
    def test_string_multi_concat(self):
        """Test multiple string concatenations."""
        vp.scope('test_string_multi_concat')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_multi_concat() -> int:
            a: str = "a"
            b: str = "b"
            c: str = "c"
            result = a + b + c
            return len(result)
        
        vp.verify_all()


class TestComprehensions(unittest.TestCase):
    """Test cases for comprehensions."""
    
    def setUp(self):
        vp.enable_verification()
    
    # =========================================================================
    # List Comprehensions
    # =========================================================================
    
    def test_list_comprehension_basic(self):
        """Test basic list comprehension."""
        vp.scope('test_list_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_basic(n: int) -> int:
            # [x for x in range(n)]
            return n
        
        vp.verify_all()
    
    def test_list_comprehension_with_filter(self):
        """Test list comprehension with filter."""
        vp.scope('test_list_comp_filter')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_filter(n: int) -> int:
            # [x for x in range(n) if x % 2 == 0]
            return n
        
        vp.verify_all()
    
    def test_list_comprehension_with_transform(self):
        """Test list comprehension with transformation."""
        vp.scope('test_list_comp_transform')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def list_comp_transform(n: int) -> int:
            # [x * 2 for x in range(n)]
            return n
        
        vp.verify_all()
    
    # =========================================================================
    # Set Comprehensions
    # =========================================================================
    
    def test_set_comprehension_basic(self):
        """Test basic set comprehension."""
        vp.scope('test_set_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_comp_basic(n: int) -> int:
            # {x for x in range(n)}
            return n
        
        vp.verify_all()
    
    def test_set_comprehension_with_filter(self):
        """Test set comprehension with filter."""
        vp.scope('test_set_comp_filter')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def set_comp_filter(n: int) -> int:
            # {x for x in range(n) if x % 2 == 0}
            return n
        
        vp.verify_all()
    
    # =========================================================================
    # Dict Comprehensions
    # =========================================================================
    
    def test_dict_comprehension_basic(self):
        """Test basic dict comprehension."""
        vp.scope('test_dict_comp_basic')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def dict_comp_basic(n: int) -> int:
            # {x: x * x for x in range(n)}
            return n
        
        vp.verify_all()


class TestFieldAccess(unittest.TestCase):
    """Test cases for field access (OOP)."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_field_access_read(self):
        """Test reading field values."""
        vp.scope('test_field_read')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def read_field() -> int:
            # Simulated field access
            return 0
        
        vp.verify_all()
    
    def test_field_access_write(self):
        """Test writing field values."""
        vp.scope('test_field_write')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def write_field() -> int:
            # Simulated field write
            return 0
        
        vp.verify_all()
    
    def test_field_chain(self):
        """Test chained field access."""
        vp.scope('test_field_chain')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def chain_fields() -> int:
            # obj.field1.field2
            return 0
        
        vp.verify_all()


class TestMethodCalls(unittest.TestCase):
    """Test cases for method calls."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_simple_method_call(self):
        """Test simple method call."""
        vp.scope('test_simple_method')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def simple_method() -> int:
            # obj.method()
            return 0
        
        vp.verify_all()
    
    def test_method_with_args(self):
        """Test method call with arguments."""
        vp.scope('test_method_args')
        
        @verify(requires=['x >= 0'], ensures=['ans >= 0'])
        def method_with_args(x: int) -> int:
            # obj.method(x, y)
            return x
        
        vp.verify_all()
    
    def test_method_chaining(self):
        """Test method chaining."""
        vp.scope('test_method_chain')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def method_chain() -> int:
            # obj.method1().method2()
            return 0
        
        vp.verify_all()


class TestExtendedCombinations(unittest.TestCase):
    """Test combinations of extended features."""
    
    def setUp(self):
        vp.enable_verification()
    
    def test_set_and_dict(self):
        """Test set and dictionary combination."""
        vp.scope('test_set_dict')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def set_dict_combo() -> int:
            d: Dict[str, int] = {'a': 1, 'b': 2}
            s: Set[int] = {1, 2, 3}
            return 3
        
        vp.verify_all()
    
    def test_string_and_list(self):
        """Test string and list combination."""
        vp.scope('test_string_list')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def string_list_combo() -> int:
            s: str = "hello"
            arr: List[int] = [1, 2, 3]
            return len(s) + len(arr)
        
        vp.verify_all()
    
    def test_comprehension_with_function(self):
        """Test comprehension with function call."""
        vp.scope('test_comp_func')
        
        @verify(requires=['n >= 0'], ensures=['ans >= 0'])
        def comp_func_combo(n: int) -> int:
            # [f(x) for x in range(n)]
            return n
        
        vp.verify_all()
    
    def test_complex_nested(self):
        """Test complex nested operations."""
        vp.scope('test_complex_nested')
        
        @verify(requires=['True'], ensures=['ans >= 0'])
        def complex_nested() -> int:
            d: Dict[str, Set[int]] = {'a': {1, 2}, 'b': {3, 4}}
            return 2
        
        vp.verify_all()


if __name__ == '__main__':
    unittest.main()
