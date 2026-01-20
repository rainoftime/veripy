"""
Tests for the auto-active verification engine.
"""

import unittest
from veripy.auto_active import (
    AutoActiveEngine,
    InferenceStrategy
)


class TestAutoActiveEngine(unittest.TestCase):
    """Test cases for the auto-active verification engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = AutoActiveEngine(strategy=InferenceStrategy.SIMPLE)
    
    def test_infer_bounds_invariant(self):
        """Test inference of bounds invariants for loops."""
        loop_info = {
            "loop_var": "i",
            "init": 0,
            "condition": "i < n",
            "body": [
                {"type": "assign", "var": "i", "expr": "i + 1"}
            ]
        }
        
        invariants = self.engine.infer_loop_invariants(loop_info)
        
        # Should infer basic bounds
        self.assertIsInstance(invariants, list)
        self.assertGreater(len(invariants), 0)
        
        # Check for variable bounds
        found_i_bounds = any("i >=" in inv or "i <=" in inv for inv in invariants)
        self.assertTrue(found_i_bounds, f"Expected bounds for i, got: {invariants}")
    
    def test_infer_type_invariant(self):
        """Test inference of type-based invariants."""
        constraints = self.engine.infer_type_constraints("x", int)
        
        self.assertIsInstance(constraints, list)
        # Should have at least type constraint
        self.assertGreater(len(constraints), 0)
    
    def test_infer_arithmetic_lemmas(self):
        """Test generation of arithmetic lemmas."""
        lemmas = self.engine.infer_arithmetic_lemmas("x + y")
        
        self.assertIsInstance(lemmas, list)
        
        # Should generate commutativity lemma
        commutativity_found = any("commutativity" in lemma.lower() or 
                                  "x + y == y + x" in lemma 
                                  for lemma in lemmas)
        self.assertTrue(commutativity_found, f"Expected commutativity lemma, got: {lemmas}")
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        initial_stats = self.engine.get_statistics()
        
        # Generate some lemmas
        self.engine.infer_arithmetic_lemmas("x + y")
        
        updated_stats = self.engine.get_statistics()
        
        # Should have generated at least one lemma
        self.assertGreater(updated_stats["lemmas_generated"], 
                          initial_stats["lemmas_generated"])
    
    def test_cache_functionality(self):
        """Test caching of inferred invariants."""
        loop_info = {
            "loop_var": "i",
            "init": 0,
            "condition": "i < n",
            "body": []
        }
        
        # First call
        invariants1 = self.engine.infer_loop_invariants(loop_info)
        
        # Should be in cache now
        self.assertGreater(len(self.engine.cache), 0)
        
        # Second call should use cache
        invariants2 = self.engine.infer_loop_invariants(loop_info)
        
        self.assertEqual(invariants1, invariants2)
    
    def test_aggressive_strategy_inference(self):
        """Test more aggressive inference strategy."""
        engine_aggressive = AutoActiveEngine(strategy=InferenceStrategy.AGGRESSIVE)
        
        loop_info = {
            "loop_var": "i",
            "init": 0,
            "condition": "i < n",
            "body": [
                {"type": "assign", "var": "j", "expr": "i + 1"}
            ]
        }
        
        invariants = engine_aggressive.infer_loop_invariants(loop_info)
        
        # Should have more invariants than simple strategy
        engine_simple = AutoActiveEngine(strategy=InferenceStrategy.SIMPLE)
        simple_invariants = engine_simple.infer_loop_invariants(loop_info)
        
        self.assertGreaterEqual(len(invariants), len(simple_invariants))


if __name__ == '__main__':
    unittest.main()
