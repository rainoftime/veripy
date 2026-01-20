"""
Test cases for Python decorators (@property, @staticmethod, @classmethod, decorator composition).
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


class TestPropertyDecorator(unittest.TestCase):
    """Test property decorator support."""
    
    def test_property_getter(self):
        """Test @property decorator."""
        translator = ExprTranslator()
        code = "@property\ndef x(self):\n    return self._x"
        tree = ast.parse(code, mode='exec')
        # Property decorator is applied to function def
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Property)
        self.assertEqual(result.name, 'x')
        self.assertTrue(result.is_getter)
    
    def test_property_setter(self):
        """Test @x.setter decorator."""
        translator = ExprTranslator()
        code = "@x.setter\ndef x(self, value):\n    self._x = value"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Property)
        self.assertEqual(result.name, 'x')
        self.assertTrue(result.is_setter)


class TestStaticMethod(unittest.TestCase):
    """Test static method decorator support."""
    
    def test_staticmethod(self):
        """Test @staticmethod decorator."""
        translator = ExprTranslator()
        code = "@staticmethod\ndef helper(x, y):\n    return x + y"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, StaticMethod)
        self.assertEqual(result.func_name, 'helper')


class TestClassMethod(unittest.TestCase):
    """Test class method decorator support."""
    
    def test_classmethod(self):
        """Test @classmethod decorator."""
        translator = ExprTranslator()
        code = "@classmethod\ndef create(cls, x):\n    return cls(x)"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ClassMethod)
        self.assertEqual(result.func_name, 'create')


class TestDecoratorComposition(unittest.TestCase):
    """Test decorator composition support."""
    
    def test_single_decorator(self):
        """Test single decorator."""
        translator = ExprTranslator()
        code = "@decorator\ndef func():\n    pass"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Decorator)
        self.assertEqual(result.name, 'decorator')
    
    def test_decorator_with_args(self):
        """Test decorator with arguments."""
        translator = ExprTranslator()
        code = "@decorator(arg1, arg2)\ndef func():\n    pass"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Decorator)
        self.assertEqual(result.name, 'decorator')
        self.assertEqual(len(result.args), 2)


if __name__ == "__main__":
    unittest.main()
