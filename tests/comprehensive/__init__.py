"""
Comprehensive Test Suite for Veripy

This package contains comprehensive test cases for veripy, organized into three main modules:

1. test_python_features.py - Tests for Python language features
   - Control flow (if/else, while, for loops)
   - Data types (int, bool, lists, sets, dicts)
   - Expressions (arithmetic, boolean, comparisons)
   - Functions (definition, calls, recursion)
   - Statements (assign, return, assert)

2. test_auto_active_features.py - Tests for auto-active verification features
   - Loop invariants (manual and automatic)
   - Quantifiers (forall, exists)
   - Termination checking (decreases clauses)
   - Frame conditions
   - Complex verification scenarios

3. test_extended_features.py - Tests for extended Python features
   - Set operations (union, intersection, membership)
   - Dictionary operations (get, keys, values)
   - String operations (concat, length, index)
   - Comprehensions (list, set, dict)
   - Field access and method calls

## Running Tests

Run all comprehensive tests:
```bash
python -m pytest tests/comprehensive/ -v
```

Run specific test file:
```bash
python -m pytest tests/comprehensive/test_python_features.py -v
```

Run specific test class:
```bash
python -m pytest tests/comprehensive/test_python_features.py::TestControlFlow -v
```

Run specific test:
```bash
python -m pytest tests/comprehensive/test_python_features.py::TestControlFlow::test_simple_if -v
```

## Test Categories

Each test file follows a consistent organization:

- **Positive tests**: Cases that should verify successfully
- **Negative tests**: Cases that document known limitations
- **Edge cases**: Boundary conditions and special values
- **Complex scenarios**: Combinations of multiple features

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention: `test_<feature>_<description>`
2. Use descriptive docstrings
3. Include both positive and negative test cases
4. Test edge cases and boundary conditions
5. Update this README if adding new test categories
"""

from .test_python_features import *
from .test_auto_active_features import *
from .test_extended_features import *
from .test_dafny_verus_patterns import *

__all__ = [
    'TestControlFlow',
    'TestDataTypes',
    'TestExpressions',
    'TestFunctions',
    'TestStatements',
    'TestEdgeCases',
    'TestLoopInvariants',
    'TestQuantifiers',
    'TestTermination',
    'TestFrameConditions',
    'TestAutomaticInference',
    'TestComplexVerification',
    'TestSetOperations',
    'TestDictionaryOperations',
    'TestStringOperations',
    'TestComprehensions',
    'TestFieldAccess',
    'TestMethodCalls',
    'TestExtendedCombinations',
    'TestSortingAlgorithms',
    'TestSearchAlgorithms',
    'TestPrefixSumAndSlidingWindow',
    'TestTwoPointers',
    'TestBitManipulation',
    'TestMathematicalProperties',
    'TestArrayManipulation',
    'TestStringAlgorithms',
    'TestAdvancedQuantifiers',
]
