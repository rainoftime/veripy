"""
Recursive function verification for veripy.

This module provides comprehensive support for verifying recursive functions,
including termination proofs, decreases clause validation, and recursive
call verification.
"""

from .verifier import (
    RecursiveVerifier,
    RecursionType,
    RecursiveCall,
    DecreasesInfo,
    recursive_verifier,
    verify_recursive,
    verify_all_recursive
)

__all__ = [
    'RecursiveVerifier',
    'RecursionType',
    'RecursiveCall',
    'DecreasesInfo',
    'recursive_verifier',
    'verify_recursive',
    'verify_all_recursive'
]
