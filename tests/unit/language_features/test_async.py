"""
Tests for async/await features.
"""

import unittest
import ast
from veripy.parser.syntax import Await, AsyncFunctionDef
from veripy.core.transformer import ExprTranslator, StmtTranslator


class TestAwaitExpression(unittest.TestCase):
    """Test await expression."""
    
    def test_await(self):
        """Test await expression."""
        translator = ExprTranslator()
        code = "(await coro)"
        tree = ast.parse(code, mode='eval')
        result = translator.visit(tree.body)
        self.assertIsInstance(result, Await)


class TestAsyncFeatures(unittest.TestCase):
    """Test async/await features."""
    
    def test_async_function(self):
        """Test async function definition."""
        translator = StmtTranslator()
        code = """
async def fetch():
    await coro()
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, AsyncFunctionDef)
        self.assertEqual(result.name, 'fetch')


if __name__ == '__main__':
    unittest.main()
