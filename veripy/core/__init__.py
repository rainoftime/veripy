"""
Core verification engine for veripy.

This module contains the core verification functionality including:
- Verification condition generation
- AST transformation
- Type system
- Pretty printing utilities
- Runtime configuration
"""

from .verify import (
    verify,
    verify_func,
    assume,
    invariant,
    decreases,
    wp,
    STORE,
    VerificationStore,
    enable_verification,
    scope,
    do_verification,
    verify_all
)

from .transformer import (
    StmtTranslator,
    ExprTranslator,
    Expr2Z3,
    subst
)

from .prettyprint import pretty_print

from .runtime_config import (
    RuntimeConfig,
    get_config,
    set_config
)

__all__ = [
    # Verification
    'verify',
    'verify_func',
    'assume',
    'invariant',
    'decreases',
    'wp',
    'STORE',
    'VerificationStore',
    'enable_verification',
    'scope',
    'do_verification',
    'verify_all',
    
    # Transformers
    'StmtTranslator',
    'ExprTranslator',
    'Expr2Z3',
    'subst',
    
    # Utilities
    'pretty_print',
    
    # Configuration
    'RuntimeConfig',
    'get_config',
    'set_config'
]
