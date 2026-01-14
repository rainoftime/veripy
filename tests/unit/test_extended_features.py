"""
Tests for extended Python features in veripy.

This test suite covers:
- Classes/OOP (field access, method calls)
- Sets (union, intersection, membership, cardinality)
- Dictionaries (get, set, keys, values)
- Strings (concat, length, substring, index)
- List comprehensions
"""

import unittest
import z3
from veripy.parser.syntax import (
    # Core expressions
    Var, Literal, BinOp, UnOp, Quantification, FunctionCall,
    # New expressions
    StringLiteral, StringConcat, StringLength, StringIndex,
    StringSubstring, StringContains,
    SetLiteral, SetOp, SetCardinality, SetComprehension,
    DictLiteral, DictGet, DictSet, DictKeys, DictValues, DictContains,
    FieldAccess, MethodCall, FieldAssignStmt,
    ListComprehension, SetOps, DictOps,
    VInt, VBool, VString,
    # Statements
    Assign, If, While, Assume, Assert, Skip, Seq,
    # For testing
    ArithOps, CompOps, BoolOps
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


class TestStatements(unittest.TestCase):
    """Test statement types."""
    
    def test_field_assign_stmt(self):
        """Test field assignment statement."""
        obj = Var("my_obj")
        field_assign = FieldAssignStmt(obj, "x", Literal(VInt(5)))
        self.assertEqual(field_assign.obj, obj)
        self.assertEqual(field_assign.field, "x")
        self.assertEqual(field_assign.value, Literal(VInt(5)))
    
    def test_skip(self):
        """Test skip statement."""
        skip = Skip()
        self.assertEqual(str(skip), "(Skip)")
    
    def test_assign(self):
        """Test assignment statement."""
        var = Var("x")
        expr = Literal(VInt(42))
        assign = Assign(var, expr)
        self.assertEqual(assign.var, var)
        self.assertEqual(assign.expr, expr)
    
    def test_if(self):
        """Test if statement."""
        cond = Literal(VBool(True))
        then_stmt = Assign(Var("x"), Literal(VInt(1)))
        else_stmt = Assign(Var("y"), Literal(VInt(2)))
        if_stmt = If(cond, then_stmt, else_stmt)
        self.assertEqual(if_stmt.cond, cond)
        self.assertEqual(if_stmt.lb, then_stmt)
        self.assertEqual(if_stmt.rb, else_stmt)
    
    def test_while(self):
        """Test while statement."""
        cond = BinOp(Var("i"), CompOps.Lt, Literal(VInt(10)))
        body = Assign(Var("i"), BinOp(Var("i"), ArithOps.Add, Literal(VInt(1))))
        while_stmt = While([], cond, body)
        self.assertEqual(while_stmt.cond, cond)
        self.assertEqual(while_stmt.body, body)


class TestZ3Translation(unittest.TestCase):
    """Test Z3 translation for new expressions."""
    
    def setUp(self):
        """Set up test fixtures."""
        from veripy.core.transformer import Expr2Z3
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


class TestBuiltinFunctionTranslation(unittest.TestCase):
    """Test built-in function translation."""
    
    def setUp(self):
        from veripy.core.transformer import Expr2Z3
        self.name_dict = {
            "x": z3.Int("x"),
            "my_set": z3.Array("my_set", z3.IntSort(), z3.BoolSort()),
            "my_dict": z3.Array("my_dict", z3.IntSort(), z3.IntSort())
        }
        self.translator = Expr2Z3(self.name_dict)
    
    def test_len_function(self):
        """Test len() function translation."""
        arr = z3.Array("arr", z3.IntSort(), z3.IntSort())
        func_call = FunctionCall(Var("len"), [Literal(VInt(0))])  # placeholder
        # Actually call should be with a proper array
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
