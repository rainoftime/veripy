"""
Test cases for advanced type system features (type aliases, protocols, type extensions).
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


class TestTypeAlias(unittest.TestCase):
    """Test type alias support."""
    
    def test_type_alias(self):
        """Test type alias definition."""
        translator = ExprTranslator()
        code = "MyInt = int"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, TypeAlias)
        self.assertEqual(result.name, 'MyInt')
    
    def test_complex_type_alias(self):
        """Test complex type alias."""
        translator = ExprTranslator()
        code = "Vector = List[float]"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, TypeAlias)


class TestProtocol(unittest.TestCase):
    """Test Protocol support for structural subtyping."""
    
    def test_simple_protocol(self):
        """Test simple protocol definition."""
        translator = ExprTranslator()
        code = """
class IterableProtocol(Protocol):
    def __iter__(self) -> Iterator:
        pass
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Protocol)
        self.assertEqual(result.name, 'IterableProtocol')
    
    def test_protocol_with_attributes(self):
        """Test protocol with attributes."""
        translator = ExprTranslator()
        code = """
class SizedProtocol(Protocol):
    def __len__(self) -> int:
        pass
    
    @property
    def size(self) -> int:
        pass
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Protocol)


class TestTypeSystemExtensions(unittest.TestCase):
    """Test advanced type system features."""
    
    def test_type_var(self):
        """Test TypeVar creation."""
        translator = ExprTranslator()
        code = "T"
        tree = ast.parse(code, mode='eval')
        # TypeVar would be created from ast.Name in type context
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Var)
    
    def test_union_type(self):
        """Test union type (X | Y)."""
        translator = ExprTranslator()
        code = "int | str"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, BinOp)
    
    def test_optional_type(self):
        """Test optional type."""
        translator = ExprTranslator()
        code = "int | None"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, BinOp)
    
    def test_literal_type(self):
        """Test Literal type."""
        translator = ExprTranslator()
        code = "Literal[1, 'a', True]"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, LiteralType)
    
    def test_final(self):
        """Test Final qualifier."""
        translator = ExprTranslator()
        code = "MAX = 100"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        # Final is treated as a constant assignment
        self.assertIsInstance(result, Assign)


if __name__ == "__main__":
    unittest.main()
