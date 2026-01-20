# Unit Tests

This directory contains unit tests for the veripy verification system, organized into logical subdirectories by feature area.

## Directory Structure

### `language_features/`
Tests for Python language features and syntax:
- Comprehensions (list, dict, set)
- Lambdas
- Decorators
- Async/await
- Match statements
- Walrus operator (`:=`)
- F-strings
- Context managers
- Generators
- Imports
- Scopes
- Variable arguments
- Exceptions
- Functions

### `data_types/`
Tests for data type operations and verification:
- Arrays and array manipulation
- Dictionaries (operations and expressions)
- Sets (operations and expressions)
- Strings (operations and expressions)
- Dataclasses
- Classes
- Structures
- General data type tests

### `control_flow/`
Tests for control flow constructs:
- Loops (while, for)
- General control flow
- Iteration
- Loop invariants
- Loop control (break, continue)

### `verification/`
Tests for verification-specific features:
- Quantifiers (basic, advanced, comprehensive)
- Refinement types
- Termination checking
- Decreases clauses
- Frames and frame conditions
- Lemmas

### `expressions/`
Tests for expressions and operations:
- General expressions
- Builtin functions
- Function calls
- Method calls
- Field access
- OOP expressions
- Statements
- Augmented assignments
- Bit manipulation
- Mathematical properties

### `algorithms/`
Tests for algorithm implementations:
- Sorting algorithms
- Search algorithms
- String algorithms
- Prefix sum
- Two pointers

### `core/`
Tests for core system functionality:
- Type system
- Error reporting
- Auto-active engine
- Auto-active integration
- Z3 translation
- Builtin translation
- Syntax exports

### `integration/`
Integration tests:
- Expression integration
- Extended combinations
- Automatic inference

### `edge_cases/`
Edge cases and miscellaneous tests:
- Edge cases
- Test cases
- Prototype tests
- Counter tests

## Running Tests

To run all unit tests:
```bash
python -m pytest tests/unit
```

To run tests in a specific subdirectory:
```bash
python -m pytest tests/unit/language_features
python -m pytest tests/unit/data_types
python -m pytest tests/unit/control_flow
# etc.
```

To run a specific test file:
```bash
python -m pytest tests/unit/data_types/test_arrays.py
```

To run a specific test method:
```bash
python -m pytest tests/unit/data_types/test_arrays.py::TestArrays::test_set_first
```

## Test Structure

Each test file follows the standard unittest pattern:
- Uses `unittest.TestCase` as the base class
- Sets up verification in `setUp()` method with `vp.enable_verification()` and `vp.scope()`
- Contains individual test methods that define functions with `@verify` decorators
- Each test method calls `vp.verify_all()` to perform verification
- The test passes if verification succeeds (no exception) and fails if verification fails (exception raised)

The tests verify that the verification system correctly validates the specified preconditions, postconditions, and invariants. The core assertion is that `vp.verify_all()` completes without raising an exception, indicating that all verification conditions are satisfied.
