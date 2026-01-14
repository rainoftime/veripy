"""
Auto-active verification features for veripy.

This module provides automatic invariant inference, lemma generation,
and termination checking capabilities.
"""

from .engine import (
    AutoActiveEngine,
    InferenceStrategy,
    InvariantCandidate,
    LemmaEngine,
    TerminationChecker,
    auto_active_engine,
    lemma_engine,
    termination_checker,
    auto_infer_invariants,
    generate_arithmetic_lemmas,
    register_lemma,
    verify_lemma,
    check_termination,
    assert_by
)

__all__ = [
    # Engine
    'AutoActiveEngine',
    'InferenceStrategy',
    'InvariantCandidate',
    
    # Lemma
    'LemmaEngine',
    'lemma_engine',
    'register_lemma',
    'verify_lemma',
    
    # Termination
    'TerminationChecker',
    'termination_checker',
    'check_termination',
    
    # Utilities
    'auto_active_engine',
    'auto_infer_invariants',
    'generate_arithmetic_lemmas',
    'assert_by'
]
