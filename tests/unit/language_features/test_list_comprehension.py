"""
Tests for list comprehension AST nodes.
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, ListComprehension, BinOp, CompOps, VInt
)


class TestListComprehension(unittest.TestCase):
    """Test list comprehension support."""
    
    def test_list_comprehension(self):
        """Test list comprehension creation."""
        element = Var("x")
        iterable = Var("arr")
        pred = BinOp(element, CompOps.Gt, Literal(VInt(0)))
        comp = ListComprehension(element, element, iterable, pred)
        self.assertEqual(comp.element_expr, element)
        self.assertEqual(comp.element_var, element)
        self.assertEqual(comp.iterable, iterable)
        self.assertEqual(comp.predicate, pred)
    
    def test_list_comprehension_without_predicate(self):
        """Test list comprehension without predicate."""
        element = Var("x")
        iterable = Var("arr")
        comp = ListComprehension(element, element, iterable)
        self.assertIsNone(comp.predicate)
    
    def test_list_comprehension_variables(self):
        """Test variable extraction from list comprehension."""
        element = Var("x")
        iterable = Var("arr")
        comp = ListComprehension(element, element, iterable)
        vars = comp.variables()
        self.assertIn("arr", vars)
        # x should not be in variables (it's bound)


if __name__ == '__main__':
    unittest.main()
