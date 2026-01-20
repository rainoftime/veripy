"""
Tests for augmented assignment operations (+=, -=, *=, etc.).
"""

import unittest
import ast
from veripy.parser.syntax import AugAssign, ArithOps
from veripy.core.transformer import StmtTranslator


class TestAugmentedAssignment(unittest.TestCase):
    """Test augmented assignment support (+=, -=, etc.)."""
    
    def test_aug_assign_add(self):
        """Test += operation."""
        translator = StmtTranslator()
        code = "x += 1"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, AugAssign)
        self.assertEqual(result.var.name, 'x')
        self.assertEqual(result.op, ArithOps.Add)
    
    def test_aug_assign_sub(self):
        """Test -= operation."""
        translator = StmtTranslator()
        code = "x -= 5"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, AugAssign)
        self.assertEqual(result.op, ArithOps.Minus)
    
    def test_aug_assign_mult(self):
        """Test *= operation."""
        translator = StmtTranslator()
        code = "x *= 2"
        tree = ast.parse(code, mode='exec')
        result = translator.visit(tree.body[0])
        self.assertIsInstance(result, AugAssign)
        self.assertEqual(result.op, ArithOps.Mult)


if __name__ == '__main__':
    unittest.main()
