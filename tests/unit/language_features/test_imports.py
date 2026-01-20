"""
Tests for import statements.
"""

import unittest
import ast
from veripy.parser.syntax import ImportStmt, ImportFrom
from veripy.core.transformer import StmtTranslator


class TestImportStatements(unittest.TestCase):
    """Test import statements."""
    
    def test_import_module(self):
        """Test import module."""
        translator = StmtTranslator()
        code = "import os"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ImportStmt)
        self.assertEqual(result.module, 'os')
    
    def test_import_module_alias(self):
        """Test import module with alias."""
        translator = StmtTranslator()
        code = "import os as operating_system"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ImportStmt)
        self.assertEqual(result.alias, 'operating_system')
    
    def test_import_from(self):
        """Test from...import statement."""
        translator = StmtTranslator()
        code = "from os import path"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ImportFrom)
        self.assertEqual(result.module, 'os')
        self.assertEqual(result.names, ['path'])
    
    def test_import_from_multiple(self):
        """Test from...import multiple names."""
        translator = StmtTranslator()
        code = "from os import path, getcwd"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, ImportFrom)
        self.assertEqual(len(result.names), 2)


if __name__ == '__main__':
    unittest.main()
