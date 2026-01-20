"""
Tests for set expression AST nodes.
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, SetLiteral, SetOp, SetCardinality,
    SetOps, VInt
)


class TestSetExpressions(unittest.TestCase):
    """Test set expression support."""
    
    def test_set_literal(self):
        """Test set literal creation."""
        elems = [Literal(VInt(1)), Literal(VInt(2)), Literal(VInt(3))]
        s = SetLiteral(elems)
        self.assertEqual(len(s.elements), 3)
    
    def test_set_union(self):
        """Test set union operation."""
        s1 = SetLiteral([Literal(VInt(1))])
        s2 = SetLiteral([Literal(VInt(2))])
        union = SetOp(s1, SetOps.Union, s2)
        self.assertEqual(union.op, SetOps.Union)
    
    def test_set_intersection(self):
        """Test set intersection operation."""
        s1 = SetLiteral([Literal(VInt(1)), Literal(VInt(2))])
        s2 = SetLiteral([Literal(VInt(2)), Literal(VInt(3))])
        inter = SetOp(s1, SetOps.Intersection, s2)
        self.assertEqual(inter.op, SetOps.Intersection)
    
    def test_set_difference(self):
        """Test set difference operation."""
        s1 = SetLiteral([Literal(VInt(1)), Literal(VInt(2))])
        s2 = SetLiteral([Literal(VInt(2))])
        diff = SetOp(s1, SetOps.Difference, s2)
        self.assertEqual(diff.op, SetOps.Difference)
    
    def test_set_membership(self):
        """Test set membership operation."""
        elem = Literal(VInt(1))
        s = SetLiteral([Literal(VInt(1)), Literal(VInt(2))])
        member = SetOp(elem, SetOps.Member, s)
        self.assertEqual(member.op, SetOps.Member)
    
    def test_set_cardinality(self):
        """Test set cardinality."""
        s = SetLiteral([Literal(VInt(1)), Literal(VInt(2)), Literal(VInt(3))])
        card = SetCardinality(s)
        self.assertEqual(card.set_expr, s)
    
    def test_set_variables(self):
        """Test variable extraction from set expressions."""
        v = Var("x")
        s1 = SetLiteral([v])
        s2 = SetLiteral([Literal(VInt(1))])
        union = SetOp(s1, SetOps.Union, s2)
        vars = union.variables()
        self.assertIn("x", vars)


if __name__ == '__main__':
    unittest.main()
