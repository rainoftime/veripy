"""
Test cases for syntax module exports.
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
    
    def test_try_exports(self):
        """Test Try and ExceptHandler are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Try']), 'Try'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['ExceptHandler']), 'ExceptHandler'))
    
    def test_loop_exports(self):
        """Test ForLoop, Break, Continue are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['ForLoop']), 'ForLoop'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Break']), 'Break'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Continue']), 'Continue'))
    
    def test_with_exports(self):
        """Test With is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['With']), 'With'))
    
    def test_aug_assign_exports(self):
        """Test AugAssign is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['AugAssign']), 'AugAssign'))
    
    def test_raise_exports(self):
        """Test Raise is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Raise']), 'Raise'))
    
    def test_global_nonlocal_exports(self):
        """Test Global and Nonlocal are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Global']), 'Global'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Nonlocal']), 'Nonlocal'))
    
    def test_import_exports(self):
        """Test ImportStmt and ImportFrom are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['ImportStmt']), 'ImportStmt'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['ImportFrom']), 'ImportFrom'))
    
    def test_lambda_exports(self):
        """Test Lambda is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Lambda']), 'Lambda'))
    
    def test_generator_exports(self):
        """Test Yield and YieldFrom are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Yield']), 'Yield'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['YieldFrom']), 'YieldFrom'))
    
    def test_await_exports(self):
        """Test Await is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Await']), 'Await'))
    
    def test_match_exports(self):
        """Test Match and related patterns are exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Match']), 'Match'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['MatchCase']), 'MatchCase'))
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Pattern']), 'Pattern'))
    
    def test_walrus_exports(self):
        """Test Walrus is exported."""
        self.assertTrue(hasattr(__import__('veripy.parser.syntax', fromlist=['Walrus']), 'Walrus'))


if __name__ == '__main__':
    unittest.main()
