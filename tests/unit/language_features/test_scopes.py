"""
Tests for variable scope declarations (global, nonlocal).
"""

import unittest
import ast
from veripy.parser.syntax import Global, Nonlocal
from veripy.core.transformer import StmtTranslator


class TestGlobalNonlocal(unittest.TestCase):
    """Test global and nonlocal declarations."""
    
    def test_global(self):
        """Test global declaration."""
        translator = StmtTranslator()
        code = "global x, y"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Global)
        self.assertEqual(result.names, ['x', 'y'])
    
    def test_nonlocal(self):
        """Test nonlocal declaration."""
        translator = StmtTranslator()
        code = "nonlocal x, y"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Nonlocal)
        self.assertEqual(result.names, ['x', 'y'])


if __name__ == '__main__':
    unittest.main()
