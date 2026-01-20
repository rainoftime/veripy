"""
Tests for generator expressions (yield, yield from).
"""

import unittest
import ast
from veripy.parser.syntax import Yield, YieldFrom
from veripy.core.transformer import ExprTranslator


class TestGenerators(unittest.TestCase):
    """Test generator expressions (yield, yield from)."""
    
    def test_yield(self):
        """Test yield expression."""
        translator = ExprTranslator()
        code = "(yield x)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Yield)
    
    def test_yield_without_value(self):
        """Test yield without value."""
        translator = ExprTranslator()
        code = "(yield)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Yield)
        self.assertIsNone(result.value)
    
    def test_yield_from(self):
        """Test yield from expression."""
        translator = ExprTranslator()
        code = "(yield from iterable)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, YieldFrom)


if __name__ == '__main__':
    unittest.main()
