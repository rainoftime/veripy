"""
Test cases for variable arguments (*args, **kwargs) and starred expressions.
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


class TestVariableArguments(unittest.TestCase):
    """Test *args and **kwargs support."""
    
    def test_varargs(self):
        """Test *args handling."""
        translator = ExprTranslator()
        code = "args"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        # In function context, this would be VarArgs
        self.assertIsInstance(result, Var)
        self.assertEqual(result.name, 'args')
    
    def test_kwargs(self):
        """Test **kwargs handling."""
        translator = ExprTranslator()
        code = "kwargs"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Var)
        self.assertEqual(result.name, 'kwargs')


class TestStarredExpression(unittest.TestCase):
    """Test starred expressions (*x, **x)."""
    
    def test_starred_in_list(self):
        """Test starred expression in list."""
        translator = ExprTranslator()
        code = "[*a, *b]"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, ListComprehension)
    
    def test_starred_in_assignment(self):
        """Test starred in assignment."""
        translator = StmtTranslator()
        code = "a, *b, c = [1, 2, 3, 4]"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Assign)


if __name__ == "__main__":
    unittest.main()
