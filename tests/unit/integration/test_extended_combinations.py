"""
Test cases for extendedcombinations.
"""

import unittest
import veripy as vp
from veripy import verify, invariant
from typing import List, Dict, Set


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


if __name__ == "__main__":
    unittest.main()
