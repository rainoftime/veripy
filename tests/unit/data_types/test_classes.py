"""
Tests for class definitions.
"""

import unittest
import ast
from veripy.parser.syntax import ClassDef
from veripy.core.transformer import StmtTranslator


class TestClassDefinitions(unittest.TestCase):
    """Test class definition support."""
    
    def test_simple_class(self):
        """Test simple class definition."""
        translator = StmtTranslator()
        code = """
class Counter:
    def __init__(self):
        self.value = 0
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ClassDef)
        self.assertEqual(result.name, 'Counter')
        self.assertTrue(len(result.methods) > 0)
    
    def test_class_with_fields(self):
        """Test class with field definitions."""
        translator = StmtTranslator()
        code = """
class Point:
    x: int
    y: int
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ClassDef)
        self.assertTrue(len(result.fields) >= 2)
    
    def test_class_method(self):
        """Test class with methods."""
        translator = StmtTranslator()
        code = """
class Counter:
    def increment(self):
        self.value += 1
"""
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ClassDef)
        method_names = [m.name for m in result.methods]
        self.assertIn('increment', method_names)


if __name__ == '__main__':
    unittest.main()
