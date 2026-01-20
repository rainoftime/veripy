"""
Tests for context managers (with statements).
"""

import unittest
import ast
from veripy.parser.syntax import With
from veripy.core.transformer import StmtTranslator


class TestWithStatement(unittest.TestCase):
    """Test with statement for context management."""
    
    def test_with_simple(self):
        """Test simple with statement."""
        translator = StmtTranslator()
        code = """
with open('file.txt') as f:
    x = 1
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, With)
        self.assertEqual(result.item_var, 'f')


if __name__ == '__main__':
    unittest.main()
