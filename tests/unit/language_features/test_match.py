"""
Tests for match statements (Python 3.10+).
"""

import unittest
import ast
from veripy.parser.syntax import Match
from veripy.core.transformer import StmtTranslator


class TestMatchStatement(unittest.TestCase):
    """Test match statement (Python 3.10+)."""
    
    def test_match_simple(self):
        """Test simple match statement."""
        translator = StmtTranslator()
        code = """
match x:
    case 1:
        y = 2
    case 2:
        y = 3
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Match)
        self.assertEqual(len(result.cases), 2)
    
    def test_match_with_guard(self):
        """Test match with guard."""
        translator = StmtTranslator()
        code = """
match x:
    case n if n > 0:
        y = 1
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Match)
        self.assertIsNotNone(result.cases[0].guard)
    
    def test_match_pattern_capture(self):
        """Test match with capture pattern."""
        translator = StmtTranslator()
        code = """
match x:
    case n:
        y = n
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Match)
    
    def test_match_pattern_sequence(self):
        """Test match with sequence pattern."""
        translator = StmtTranslator()
        code = """
match x:
    case [a, b]:
        y = a + b
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Match)


if __name__ == '__main__':
    unittest.main()
