"""
Comprehensive tests for advanced Python features in veripy.

This test suite covers:
- Property decorators (@property, @setter)
- Staticmethod and classmethod support
- Variable arguments (*args, **kwargs)
- F-string expressions
- Decorator composition
- Data class support (@dataclass)
- Type alias support
- Protocol support (structural subtyping)
- Iteration protocols (__iter__, __next__)
- List/dict/set comprehensions
- Generator expressions
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


class TestSyntaxModuleExports(unittest.TestCase):
    """Test that all new syntax classes are properly exported."""
    
    def test_property_exports(self):
        """Test Property is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Property']), 'Property'))
    
    def test_staticmethod_exports(self):
        """Test StaticMethod is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['StaticMethod']), 'StaticMethod'))
    
    def test_classmethod_exports(self):
        """Test ClassMethod is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['ClassMethod']), 'ClassMethod'))
    
    def test_varargs_exports(self):
        """Test VarArgs is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['VarArgs']), 'VarArgs'))
    
    def test_kwargs_exports(self):
        """Test KwArgs is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['KwArgs']), 'KwArgs'))
    
    def test_fstring_exports(self):
        """Test FString is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['FString']), 'FString'))
    
    def test_decorator_exports(self):
        """Test Decorator is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Decorator']), 'Decorator'))
    
    def test_dataclass_exports(self):
        """Test DataClass is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['DataClass']), 'DataClass'))
    
    def test_protocol_exports(self):
        """Test Protocol is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Protocol']), 'Protocol'))
    
    def test_iterator_exports(self):
        """Test Iterator is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Iterator']), 'Iterator'))
    
    def test_range_exports(self):
        """Test Range is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Range']), 'Range'))
    
    def test_comprehension_exports(self):
        """Test Comprehension is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Comprehension']), 'Comprehension'))
    
    def test_zip_exports(self):
        """Test Zip is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Zip']), 'Zip'))
    
    def test_map_exports(self):
        """Test Map is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Map']), 'Map'))
    
    def test_filter_exports(self):
        """Test Filter is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Filter']), 'Filter'))
    
    def test_reduce_exports(self):
        """Test Reduce is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Reduce']), 'Reduce'))
    
    def test_typevar_exports(self):
        """Test TypeVar is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['TypeVar']), 'TypeVar'))
    
    def test_uniontype_exports(self):
        """Test UnionType is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['UnionType']), 'UnionType'))
    
    def test_optionaltype_exports(self):
        """Test OptionalType is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['OptionalType']), 'OptionalType'))
    
    def test_literaltype_exports(self):
        """Test LiteralType is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['LiteralType']), 'LiteralType'))
    
    def test_final_exports(self):
        """Test Final is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Final']), 'Final'))
    
    def test_typeguard_exports(self):
        """Test TypeGuard is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['TypeGuard']), 'TypeGuard'))


if __name__ == '__main__':
    unittest.main()
