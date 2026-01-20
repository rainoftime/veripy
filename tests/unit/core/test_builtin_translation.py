"""
Tests for built-in function translation to Z3.
"""

import unittest
import z3
from veripy.parser.syntax import Var, FunctionCall, VInt
from veripy.core.transformer import Expr2Z3


class TestBuiltinFunctionTranslation(unittest.TestCase):
    """Test built-in function translation."""
    
    def setUp(self):
        self.name_dict = {
            "x": z3.Int("x"),
            "my_set": z3.Array("my_set", z3.IntSort(), z3.BoolSort()),
            "my_dict": z3.Array("my_dict", z3.IntSort(), z3.IntSort())
        }
        self.translator = Expr2Z3(self.name_dict)
    
    def test_len_function(self):
        """Test len() function translation."""
        arr = z3.Array("arr", z3.IntSort(), z3.IntSort())
        self.name_dict["test_arr"] = arr
        func_call = FunctionCall(Var("len"), [Var("test_arr")])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.IntSort())
    
    def test_set_function(self):
        """Test set() function translation."""
        func_call = FunctionCall(Var("set"), [])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))
    
    def test_card_function(self):
        """Test card() function translation."""
        my_set = z3.Array("my_set", z3.IntSort(), z3.BoolSort())
        self.name_dict["s"] = my_set
        func_call = FunctionCall(Var("card"), [Var("s")])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.IntSort())
    
    def test_mem_function(self):
        """Test mem() function translation."""
        my_set = z3.Array("my_set", z3.IntSort(), z3.BoolSort())
        self.name_dict["s"] = my_set
        self.name_dict["x"] = z3.Int("x")
        func_call = FunctionCall(Var("mem"), [Var("x"), Var("s")])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.BoolSort())
    
    def test_dict_function(self):
        """Test dict() function translation."""
        func_call = FunctionCall(Var("dict"), [])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.ArraySort(z3.IntSort(), z3.IntSort()))
    
    def test_keys_function(self):
        """Test keys() function translation."""
        my_dict = z3.Array("my_dict", z3.IntSort(), z3.IntSort())
        self.name_dict["d"] = my_dict
        func_call = FunctionCall(Var("keys"), [Var("d")])
        result = self.translator.visit(func_call)
        self.assertEqual(result.sort(), z3.ArraySort(z3.IntSort(), z3.BoolSort()))


if __name__ == '__main__':
    unittest.main()
