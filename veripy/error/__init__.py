"""
Error reporting for veripy.

This module provides comprehensive error reporting with source locations,
counterexamples, and suggestions for fixing verification failures.
"""

from .reporter import (
    ErrorReporter,
    ErrorSeverity,
    ErrorCategory,
    SourceLocation,
    Counterexample,
    VerificationError,
    error_reporter,
    report_verification_failure,
    print_verification_report,
    extract_source_location,
    parse_smt_model
)

__all__ = [
    'ErrorReporter',
    'ErrorSeverity',
    'ErrorCategory',
    'SourceLocation',
    'Counterexample',
    'VerificationError',
    'error_reporter',
    'report_verification_failure',
    'print_verification_report',
    'extract_source_location',
    'parse_smt_model'
]
