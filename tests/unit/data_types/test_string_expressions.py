"""
Tests for string expression AST nodes.
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, StringLiteral, StringConcat, StringLength,
    StringIndex, VInt
)


class TestStringExpressions(unittest.TestCase):
    """Test string expression support."""
    
    def test_string_literal(self):
        """Test string literal creation."""
        s = StringLiteral("hello")
        self.assertEqual(s.value, "hello")
    
    def test_string_concat(self):
        """Test string concatenation."""
        s1 = StringLiteral("hello")
        s2 = StringLiteral("world")
        concat = StringConcat(s1, s2)
        self.assertEqual(concat.left, s1)
        self.assertEqual(concat.right, s2)
    
    def test_string_length(self):
        """Test string length."""
        s = StringLiteral("test")
        length = StringLength(s)
        self.assertEqual(length.string_expr, s)
    
    def test_string_index(self):
        """Test string indexing."""
        s = StringLiteral("test")
        idx = Literal(VInt(0))
        indexed = StringIndex(s, idx)
        self.assertEqual(indexed.string_expr, s)
        self.assertEqual(indexed.index, idx)
    
    def test_string_variables(self):
        """Test variable extraction from string expressions."""
        v = Var("x")
        s = StringConcat(v, StringLiteral("test"))
        vars = s.variables()
        self.assertIn("x", vars)


if __name__ == '__main__':
    unittest.main()
