"""
Tests for walrus operator (:=).
"""

import unittest
import ast
from veripy.parser.syntax import Walrus
from veripy.core.transformer import ExprTranslator


class TestWalrusOperator(unittest.TestCase):
    """Test walrus operator (:=)."""
    
    def test_walrus(self):
        """Test named expression (walrus operator)."""
        translator = ExprTranslator()
        code = "(x := 1)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Walrus)
        self.assertEqual(result.name, 'x')


if __name__ == '__main__':
    unittest.main()
