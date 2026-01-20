"""
Tests for loop control statements (break, continue, for loops).
"""

import unittest
import ast
from veripy.parser.syntax import Break, Continue, Seq, While
from veripy.core.transformer import StmtTranslator


class TestLoopStatements(unittest.TestCase):
    """Test loop control statements (break, continue, for loops)."""
    
    def test_break(self):
        """Test break statement."""
        translator = StmtTranslator()
        code = "break"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Break)
    
    def test_continue(self):
        """Test continue statement."""
        translator = StmtTranslator()
        code = "continue"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Continue)
    
    def test_for_loop_range(self):
        """Test for loop with range - transformed to While loop with initialization."""
        translator = StmtTranslator()
        code = "for i in range(10):\n    x = i"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        # For loops are transformed to Seq containing initialization and While loop
        # The Seq preserves the semantics: init; while cond: body
        self.assertIsInstance(result, Seq)
        # The Seq should contain a While loop
        self.assertIsInstance(result.s2, While)
        self.assertIn('i', str(result.s2.cond))


if __name__ == '__main__':
    unittest.main()
