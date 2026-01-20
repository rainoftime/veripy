"""
Test cases for @dataclass decorator.
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


class TestDataClass(unittest.TestCase):
    """Test data class decorator support."""
    
    def test_dataclass(self):
        """Test @dataclass decorator."""
        translator = ExprTranslator()
        code = "@dataclass\nclass Point:\n    x: int\n    y: int"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, DataClass)
        self.assertEqual(result.name, 'Point')
    
    def test_dataclass_with_options(self):
        """Test @dataclass with options."""
        translator = ExprTranslator()
        code = "@dataclass(init=False, eq=False)\nclass Data:\n    value: int"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, DataClass)
        self.assertFalse(result.init)
        self.assertFalse(result.eq)


if __name__ == "__main__":
    unittest.main()
