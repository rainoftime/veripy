"""
Tests for OOP expressions (field access, method calls).
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, FieldAccess, MethodCall, FieldAssignStmt,
    VInt
)


class TestOOPExpressions(unittest.TestCase):
    """Test OOP expression support."""
    
    def test_field_access(self):
        """Test field access expression."""
        obj = Var("my_obj")
        field = FieldAccess(obj, "x")
        self.assertEqual(field.obj, obj)
        self.assertEqual(field.field, "x")
    
    def test_method_call(self):
        """Test method call expression."""
        obj = Var("my_obj")
        method = MethodCall(obj, "compute", [Literal(VInt(1)), Literal(VInt(2))])
        self.assertEqual(method.obj, obj)
        self.assertEqual(method.method_name, "compute")
        self.assertEqual(len(method.args), 2)
    
    def test_field_access_variables(self):
        """Test variable extraction from field access."""
        obj = Var("x")
        field = FieldAccess(obj, "field")
        vars = field.variables()
        self.assertIn("x", vars)
    
    def test_method_call_variables(self):
        """Test variable extraction from method call."""
        obj = Var("obj")
        arg = Var("y")
        method = MethodCall(obj, "method", [arg])
        vars = method.variables()
        self.assertIn("obj", vars)
        self.assertIn("y", vars)
    
    def test_field_assign_stmt(self):
        """Test field assignment statement."""
        obj = Var("my_obj")
        field_assign = FieldAssignStmt(obj, "x", Literal(VInt(5)))
        self.assertEqual(field_assign.obj, obj)
        self.assertEqual(field_assign.field, "x")
        self.assertEqual(field_assign.value, Literal(VInt(5)))


if __name__ == '__main__':
    unittest.main()
