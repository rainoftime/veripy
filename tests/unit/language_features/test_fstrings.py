"""
Test cases for f-string expressions.
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


class TestFString(unittest.TestCase):
    """Test f-string expression support."""
    
    def test_fstring_simple(self):
        """Test simple f-string."""
        translator = ExprTranslator()
        code = 'f"hello {name}"'
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, FString)
        self.assertEqual(result.literal_parts, ['hello ', ''])
    
    def test_fstring_multiple(self):
        """Test f-string with multiple interpolations."""
        translator = ExprTranslator()
        code = 'f"{x} + {y} = {x + y}"'
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, FString)
        self.assertEqual(len(result.parts), 3)


if __name__ == "__main__":
    unittest.main()
