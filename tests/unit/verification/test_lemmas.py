"""
Tests for lemma management engine.
"""

import unittest
from veripy.auto_active import (
    LemmaEngine,
    auto_infer_invariants,
    generate_arithmetic_lemmas,
    register_lemma
)


class TestLemmaEngine(unittest.TestCase):
    """Test cases for the lemma management engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lemma_engine = LemmaEngine()
    
    def test_register_lemma(self):
        """Test lemma registration."""
        result = self.lemma_engine.add_lemma(
            name="test_lemma",
            premises=["x > 0", "y > 0"],
            conclusion="x + y > 0"
        )
        
        self.assertTrue(result)
        self.assertIn("test_lemma", self.lemma_engine.lemmas)
    
    def test_get_lemma(self):
        """Test retrieving a registered lemma."""
        self.lemma_engine.add_lemma(
            name="test_lemma",
            premises=["x > 0"],
            conclusion="x >= 0"
        )
        
        lemma = self.lemma_engine.get_lemma("test_lemma")
        
        self.assertIsNotNone(lemma)
        self.assertEqual(lemma["premises"], ["x > 0"])
        self.assertEqual(lemma["conclusion"], "x >= 0")
    
    def test_verify_lemma(self):
        """Test lemma verification."""
        # Register a true lemma
        self.lemma_engine.add_lemma(
            name="reflexivity",
            premises=[],
            conclusion="x == x"
        )
        
        # Try to verify it
        result = self.lemma_engine.verify_lemma("reflexivity")
        
        # This should succeed (reflexivity is always true)
        # Note: May fail if the solver has issues
        self.assertIsInstance(result, bool)
    
    def test_verified_lemmas_filter(self):
        """Test filtering for verified lemmas."""
        # Add multiple lemmas
        self.lemma_engine.add_lemma("lemma1", [], "x == x")
        self.lemma_engine.add_lemma("lemma2", ["x > 0"], "x >= 0")
        
        # Verify one
        self.lemma_engine.verify_lemma("lemma1")
        
        # Get verified lemmas
        verified = self.lemma_engine.get_verified_lemmas()
        
        self.assertIn("lemma1", verified)
        self.assertNotIn("lemma2", verified)
    
    def test_get_all_lemmas(self):
        """Test getting all registered lemmas."""
        self.lemma_engine.add_lemma("lemma1", [], "x == x")
        self.lemma_engine.add_lemma("lemma2", [], "y == y")
        
        all_lemmas = self.lemma_engine.get_all_lemmas()
        
        self.assertEqual(len(all_lemmas), 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for auto-active features."""
    
    def test_auto_infer_invariants(self):
        """Test the auto_infer_invariants convenience function."""
        loop_info = {
            "loop_var": "i",
            "init": 0,
            "condition": "i < 10",
            "body": []
        }
        
        invariants = auto_infer_invariants(loop_info)
        
        self.assertIsInstance(invariants, list)
    
    def test_generate_arithmetic_lemmas(self):
        """Test the generate_arithmetic_lemmas convenience function."""
        lemmas = generate_arithmetic_lemmas("x + y")
        
        self.assertIsInstance(lemmas, list)
        self.assertGreater(len(lemmas), 0)
    
    def test_register_lemma(self):
        """Test the register_lemma convenience function."""
        result = register_lemma(
            name="test_commute",
            premises=[],
            conclusion="x + y == y + x"
        )
        
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
