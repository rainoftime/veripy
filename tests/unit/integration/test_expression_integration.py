"""
Integration tests combining multiple expression features.
"""

import unittest
from veripy.parser.syntax import (
    Var, Literal, StringLiteral, StringConcat, StringLength,
    SetLiteral, SetOp, SetOps, DictLiteral, DictGet, VInt
)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features."""
    
    def test_set_operations_chain(self):
        """Test chaining set operations."""
        s1 = SetLiteral([Literal(VInt(1)), Literal(VInt(2))])
        s2 = SetLiteral([Literal(VInt(2)), Literal(VInt(3))])
        s3 = SetLiteral([Literal(VInt(3)), Literal(VInt(4))])
        
        # (s1 union s2) intersection s3
        union = SetOp(s1, SetOps.Union, s2)
        inter = SetOp(union, SetOps.Intersection, s3)
        
        vars = inter.variables()
        self.assertEqual(len(vars), 0)  # No variables
    
    def test_dict_with_string_keys(self):
        """Test dictionary with string keys."""
        d = DictLiteral(
            [StringLiteral("a"), StringLiteral("b")],
            [Literal(VInt(1)), Literal(VInt(2))]
        )
        
        # Get operation
        get = DictGet(d, StringLiteral("a"))
        vars = get.variables()
        self.assertEqual(len(vars), 0)
    
    def test_string_operations(self):
        """Test string operations chain."""
        s1 = StringLiteral("hello")
        s2 = StringLiteral(" ")
        s3 = StringLiteral("world")
        
        concat = StringConcat(StringConcat(s1, s2), s3)
        length = StringLength(concat)
        
        vars = length.variables()
        self.assertEqual(len(vars), 0)


if __name__ == '__main__':
    unittest.main()
