"""
Test cases for builtin functions.
"""

import unittest
import ast
from veripy.parser.syntax import (
    # Core expressions
    Var, Literal, BinOp, UnOp, Quantification, FunctionCall,
    # Advanced features
    Property, StaticMethod, ClassMethod, VarArgs, KwArgs,
    FString, Decorator, DecoratorChain, DataClass, TypeAlias,
    Protocol, MethodSignature, Iterator, Range, Enumerate, Zip,
    Map, Filter, Reduce, Comprehension, Generator,
    TypeVar, UnionType, OptionalType, LiteralType, Final, TypeGuard,
    ListComprehension, SetComprehension, DictLiteral,
    # Statements
    Assign, If, While, Assume, Assert, Skip, Seq, ClassDef, MethodDef,
    # Values
    VInt, VBool, VString,
    # Operations
    ArithOps, CompOps, BoolOps
)
from veripy.core.transformer import ExprTranslator, StmtTranslator


class TestAdvancedBuiltins(unittest.TestCase):
    """Test advanced built-in functions."""
    
    def test_zip(self):
        """Test zip function."""
        translator = ExprTranslator()
        code = "zip(a, b)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Zip)
    
    def test_map(self):
        """Test map function."""
        translator = ExprTranslator()
        code = "map(f, items)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Map)
    
    def test_filter(self):
        """Test filter function."""
        translator = ExprTranslator()
        code = "filter(pred, items)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Filter)
    
    def test_reduce(self):
        """Test reduce function."""
        translator = ExprTranslator()
        code = "reduce(func, items, initial)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Reduce)


if __name__ == '__main__':
    unittest.main()
