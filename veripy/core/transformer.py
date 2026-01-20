"""
Extended transformer for veripy.

This module adds support for:
- Classes/OOP (field access, method calls)
- Sets (union, intersection, membership, cardinality)
- Dictionaries (get, set, keys, values)
- Strings (concat, length, substring, index)
- List comprehensions (transformed to loops)

This file now imports from the transformer subdirectory for backward compatibility.
All functionality has been split into separate modules:
- transformer/utils.py: Utility functions
- transformer/expr_translator.py: Expression translator
- transformer/stmt_translator.py: Statement translator
- transformer/z3_translator.py: Z3 translator
"""

# Import everything from the transformer subdirectory for backward compatibility
from .transformer import (
    ExprTranslator,
    StmtTranslator,
    Expr2Z3,
    subst,
    raise_exception,
)

__all__ = [
    'ExprTranslator',
    'StmtTranslator',
    'Expr2Z3',
    'subst',
    'raise_exception',
]
