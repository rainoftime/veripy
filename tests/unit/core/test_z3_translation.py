"""
Tests for Z3 translation of extended expressions.
"""

import unittest
import z3
from veripy.parser.syntax import (
    Var, Literal, StringLiteral, StringConcat,
    SetLiteral, SetOp, SetOps, DictLiteral,
    FieldAccess, VInt
)
from veripy.core.transformer import Expr2Z3


class TestZ3Translation(unittest.TestCase):
    """Test Z3 translation for new expressions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.name_dict = {
            "x": z3.Int("x"),
            "y": z3.Int("y"),
            "s": z3.String("s"),
            "arr": z3.Array("arr", z3.IntSort(), z3.IntSort())
        }
        self.translator = Expr2Z3(self.name_dict)
    
    def test_translate_string_literal(self):
        """Test translating string literal to Z3."""
        s = StringLiteral("hello")
        result = self.translator.visit(s)
        self.assertEqual(result, z3.StringVal("hello"))
    
    def test_translate_string_concat(self):
        """Test translating string concatenation to Z3."""
        s1 = StringLiteral("hello")
        s2 = StringLiteral("world")
        concat = StringConcat(s1, s2)
        result = self.translator.visit(concat)
        # Result should be an application of str_concat
        self.assertIsInstance(result, z3.ExprRef)
        self.assertEqual(result.decl().name(), "str_concat")
    
    def test_translate_set_literal(self):
        """Test translating set literal to Z3."""
        elems = [Literal(VInt(1)), Literal(VInt(2))]
        s = SetLiteral(elems)
        result = self.translator.visit(s)
        # Should be a Z3 array representing the set
        self.assertEqual(result.sort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    
    def test_translate_set_op(self):
        """Test translating set operation to Z3."""
        s1 = SetLiteral([Literal(VInt(1))])
        s2 = SetLiteral([Literal(VInt(2))])
        union = SetOp(s1, SetOps.Union, s2)
        result = self.translator.visit(union)
        self.assertEqual(result.sort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    
    def test_translate_dict_literal(self):
        """Test translating dict literal to Z3."""
        keys = [StringLiteral("a"), StringLiteral("b")]
        values = [Literal(VInt(1)), Literal(VInt(2))]
        d = DictLiteral(keys, values)
        result = self.translator.visit(d)
        # Should be a Z3 array representing the dict
        self.assertEqual(result.sort(), z3.ArraySort(z3.StringSort(), z3.IntSort()))
    
    def test_translate_field_access(self):
        """Test translating field access to Z3."""
        obj = Var("my_obj")
        field = FieldAccess(obj, "x")
        # This will create an uninterpreted function
        result = self.translator.visit(field)
        # Should be a function application
        self.assertIsInstance(result, z3.ExprRef)


if __name__ == '__main__':
    unittest.main()
