"""
Tests for dictionary expression AST nodes.
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, StringLiteral, DictLiteral, DictGet, DictSet,
    DictKeys, DictContains, VInt
)


class TestDictExpressions(unittest.TestCase):
    """Test dictionary expression support."""
    
    def test_dict_literal(self):
        """Test dict literal creation."""
        keys = [StringLiteral("a"), StringLiteral("b")]
        values = [Literal(VInt(1)), Literal(VInt(2))]
        d = DictLiteral(keys, values)
        self.assertEqual(len(d.keys), 2)
        self.assertEqual(len(d.values), 2)
    
    def test_dict_get(self):
        """Test dict get operation."""
        d = Var("my_dict")
        key = StringLiteral("key")
        get = DictGet(d, key)
        self.assertEqual(get.dict_expr, d)
        self.assertEqual(get.key, key)
    
    def test_dict_get_with_default(self):
        """Test dict get with default value."""
        d = Var("my_dict")
        key = StringLiteral("key")
        default = Literal(VInt(0))
        get = DictGet(d, key, default)
        self.assertEqual(get.default, default)
    
    def test_dict_set(self):
        """Test dict set operation."""
        d = Var("my_dict")
        key = StringLiteral("key")
        value = Literal(VInt(42))
        new_dict = DictSet(d, key, value)
        self.assertEqual(new_dict.dict_expr, d)
        self.assertEqual(new_dict.key, key)
    
    def test_dict_keys(self):
        """Test dict keys operation."""
        d = Var("my_dict")
        keys = DictKeys(d)
        self.assertEqual(keys.dict_expr, d)
    
    def test_dict_contains(self):
        """Test dict contains operation."""
        d = Var("my_dict")
        key = StringLiteral("key")
        contains = DictContains(d, key)
        self.assertEqual(contains.dict_expr, d)
    
    def test_dict_variables(self):
        """Test variable extraction from dict expressions."""
        d = Var("dict_var")
        key = Var("k")
        get = DictGet(d, key)
        vars = get.variables()
        self.assertIn("dict_var", vars)
        self.assertIn("k", vars)


if __name__ == '__main__':
    unittest.main()
