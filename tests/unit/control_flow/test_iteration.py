"""
Test cases for iteration protocols and comprehensions.
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


class TestIterationProtocol(unittest.TestCase):
    """Test iteration protocol support."""
    
    def test_iterator(self):
        """Test iterator expression."""
        translator = ExprTranslator()
        code = "iter([1, 2, 3])"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, FunctionCall)
    
    def test_range(self):
        """Test range expression."""
        translator = ExprTranslator()
        code = "range(10)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Range)
    
    def test_range_with_start_stop(self):
        """Test range with start and stop."""
        translator = ExprTranslator()
        code = "range(1, 10)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Range)
    
    def test_range_with_step(self):
        """Test range with step."""
        translator = ExprTranslator()
        code = "range(0, 10, 2)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Range)
    
    def test_enumerate(self):
        """Test enumerate expression."""
        translator = ExprTranslator()
        code = "enumerate(items)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Enumerate)
    
    def test_enumerate_with_start(self):
        """Test enumerate with start."""
        translator = ExprTranslator()
        code = "enumerate(items, 1)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Enumerate)


class TestComprehensions(unittest.TestCase):
    """Test list/dict/set comprehensions and generator expressions."""
    
    def test_list_comprehension(self):
        """Test list comprehension."""
        translator = ExprTranslator()
        code = "[x for x in range(10)]"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, ListComprehension)
    
    def test_list_comprehension_with_filter(self):
        """Test list comprehension with filter."""
        translator = ExprTranslator()
        code = "[x for x in range(10) if x % 2 == 0]"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, ListComprehension)
        self.assertIsNotNone(result.predicate)
    
    def test_set_comprehension(self):
        """Test set comprehension."""
        translator = ExprTranslator()
        code = "{x for x in range(10)}"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, SetComprehension)
    
    def test_dict_comprehension(self):
        """Test dict comprehension."""
        translator = ExprTranslator()
        code = "{k: v for k, v in items}"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, DictLiteral)  # Simplified
    
    def test_generator_expression(self):
        """Test generator expression."""
        translator = ExprTranslator()
        code = "(x for x in range(10))"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Comprehension)


if __name__ == "__main__":
    unittest.main()
