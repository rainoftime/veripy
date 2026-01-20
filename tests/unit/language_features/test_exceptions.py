"""
Tests for exception handling (try/except/finally) and raise statements.
"""

import unittest
import ast
from veripy.parser.syntax import Try, ExceptHandler, Raise
from veripy.core.transformer import StmtTranslator


class TestExceptionHandling(unittest.TestCase):
    """Test exception handling (try/except/finally/else)."""
    
    def test_try_except(self):
        """Test simple try/except."""
        translator = StmtTranslator()
        code = """
try:
    x = 1
except ValueError:
    x = 2
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Try)
        self.assertEqual(len(result.handlers), 1)
        self.assertEqual(result.handlers[0].exc_type, 'ValueError')
    
    def test_try_except_as(self):
        """Test try/except with variable binding."""
        translator = StmtTranslator()
        code = """
try:
    x = 1
except Exception as e:
    x = 2
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Try)
        self.assertEqual(result.handlers[0].exc_var, 'e')
    
    def test_try_except_else_finally(self):
        """Test try/except/else/finally."""
        translator = StmtTranslator()
        code = """
try:
    x = 1
except:
    x = 2
else:
    x = 3
finally:
    x = 4
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Try)
        self.assertIsNotNone(result.orelse)
        self.assertIsNotNone(result.finalbody)
    
    def test_multiple_handlers(self):
        """Test multiple except handlers."""
        translator = StmtTranslator()
        code = """
try:
    x = 1
except ValueError:
    x = 2
except TypeError:
    x = 3
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Try)
        self.assertEqual(len(result.handlers), 2)


class TestRaiseStatement(unittest.TestCase):
    """Test raise statement for exceptions."""
    
    def test_raise_simple(self):
        """Test simple raise."""
        translator = StmtTranslator()
        code = "raise ValueError()"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Raise)
    
    def test_raise_with_message(self):
        """Test raise with message."""
        translator = StmtTranslator()
        code = "raise ValueError('error message')"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Raise)
    
    def test_raise_from(self):
        """Test raise ... from ... statement."""
        translator = StmtTranslator()
        code = "raise ValueError() from e"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, Raise)
        self.assertIsNotNone(result.cause)


if __name__ == '__main__':
    unittest.main()
