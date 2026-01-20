"""
Tests for lambda expressions.
"""

import unittest
import ast
from veripy.parser.syntax import Lambda
from veripy.core.transformer import ExprTranslator


class TestLambdaExpression(unittest.TestCase):
    """Test lambda expressions."""
    
    def test_lambda_simple(self):
        """Test simple lambda."""
        translator = ExprTranslator()
        code = "lambda x: x + 1"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Lambda)
        self.assertEqual(len(result.params), 1)
    
    def test_lambda_multiple_params(self):
        """Test lambda with multiple parameters."""
        translator = ExprTranslator()
        code = "lambda x, y: x + y"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Lambda)
        self.assertEqual(len(result.params), 2)


if __name__ == '__main__':
    unittest.main()
