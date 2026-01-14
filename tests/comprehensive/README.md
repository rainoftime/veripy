# Comprehensive Test Suite for Veripy

This directory contains comprehensive test cases for veripy, designed to guide the improvement of the auto-active verification system.

## Test Coverage Overview

The test suite is organized into four main modules covering:

### 1. Python Features (`test_python_features.py`)
- **TestControlFlow**: If/else, while loops, for loops, nested control flow
- **TestDataTypes**: Integers, booleans, lists, sets, dictionaries
- **TestExpressions**: Arithmetic, unary, boolean, comparisons, membership
- **TestFunctions**: Basic calls, recursion, type annotations
- **TestStatements**: Assignment, return, assert, pass
- **TestEdgeCases**: Zero values, negatives, large values, edge conditions

### 2. Auto-Active Features (`test_auto_active_features.py`)
- **TestLoopInvariants**: Manual invariants, accumulator patterns, nested loops
- **TestQuantifiers**: forall, exists, quantifier combinations
- **TestTermination**: decreases clauses, termination patterns
- **TestFrameConditions**: Variable modification tracking
- **TestAutomaticInference**: Auto-inferred bounds and relationships
- **TestComplexVerification**: Binary search, merge sort, DP, GCD

### 3. Extended Features (`test_extended_features.py`)
- **TestSetOperations**: Union, intersection, difference, membership, cardinality
- **TestDictionaryOperations**: Get, keys, values, contains
- **TestStringOperations**: Concat, length, index, contains
- **TestComprehensions**: List, set, dict comprehensions
- **TestFieldAccess**: Object field reading/writing
- **TestMethodCalls**: Method invocation and chaining
- **TestExtendedCombinations**: Feature combinations

### 4. Dafny & Verus Patterns (`test_dafny_verus_patterns.py`)
Advanced test cases inspired by Dafny and Verus verification systems:

**Sorting Algorithms:**
- Bubble sort with permutation invariant
- Selection sort with min-finding
- Insertion sort with sorted prefix
- Merge sort with recursive structure

**Search Algorithms:**
- Binary search (classic and variant)
- Linear search with counting
- Two-sum on sorted array

**Two Pointers:**
- Remove duplicates
- Reverse array
- Palindrome check
- Merge sorted arrays
- Trapping rain water

**Prefix Sum & Sliding Window:**
- Basic prefix sum
- Range query
- Sliding window sum
- Subarray sum to k

**Bit Manipulation:**
- Count set bits (basic and Kernighan's)
- Power of two check
- Bit parity
- XOR properties
- Range bitwise AND

**Mathematical Properties:**
- GCD with Euclidean algorithm
- Fibonacci (recursive and iterative)
- Power with exponentiation by squaring
- Summation formulas

**String Algorithms:**
- Anagram check
- Longest palindrome
- Substring search

**Advanced Quantifiers:**
- Sorted property with forall
- Existence search
- Uniqueness check
- Min/max with quantifiers

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/comprehensive/ -v
```

### Run Specific Module
```bash
python -m pytest tests/comprehensive/test_python_features.py -v
python -m pytest tests/comprehensive/test_dafny_verus_patterns.py -v
```

### Run Specific Test Class
```bash
python -m pytest tests/comprehensive/test_python_features.py::TestControlFlow -v
python -m pytest tests/comprehensive/test_dafny_verus_patterns.py::TestSortingAlgorithms -v
```

### Run Specific Test
```bash
python -m pytest tests/comprehensive/test_python_features.py::TestControlFlow::test_simple_if -v
```

## Test Statistics

| Module | Test Classes | Test Methods |
|--------|-------------|--------------|
| Python Features | 6 | 61 |
| Auto-Active Features | 6 | 42 |
| Extended Features | 7 | 42 |
| Dafny & Verus Patterns | 9 | 56 |
| **Total** | **28** | **201** |

## Adding New Tests

### Naming Conventions
- Test files: `test_<category>.py`
- Test classes: `Test<CamelCase>`
- Test methods: `test_<snake_case>`

### Template for New Tests

```python
def test_feature_description(self):
    """Brief description of what this test verifies."""
    vp.scope('test_unique_name')
    
    @verify(requires=['precondition'], ensures=['postcondition'])
    def function_to_verify(arg: type) -> return_type:
        # Implementation
        return result
    
    vp.verify_all()
```

### Best Practices
1. Use descriptive test names
2. Include clear preconditions and postconditions
3. Test both positive and negative cases
4. Cover edge cases and boundary conditions
5. Use `vp.invariant()` for loop verification
6. Use `vp.scope()` to isolate test cases

## Coverage Areas

### Currently Covered
- Basic control flow structures
- Arithmetic and boolean expressions
- Recursive functions with termination
- Loop invariants (manual)
- Quantifiers (forall, exists)
- Basic data structures (lists, sets, dicts)
- String operations
- Sorting algorithms (bubble, selection, insertion, merge)
- Search algorithms (linear, binary)
- Two pointers technique
- Prefix sums and sliding window
- Bit manipulation (count, parity, XOR)
- Mathematical functions (gcd, fibonacci, power)
- String algorithms (anagram, palindrome)

### Not Yet Covered (Future Work)
- Exception handling verification
- Class inheritance verification
- Async/await verification
- Generator verification
- Complex frame conditions
- Custom lemma generation
- Counterexample generation
- Heap sort and quick sort
- Graph algorithms (DFS, BFS, shortest path)

## Integration with CI/CD

The tests can be integrated into CI pipelines:

```yaml
# GitHub Actions example
- name: Run Comprehensive Tests
  run: python -m pytest tests/comprehensive/ -v --tb=short
```

## Related Documentation

- [FEATURE_SUPPORT.md](../../docs/FEATURE_SUPPORT.md) - Full feature support matrix
- [EXTENDED_FEATURES.md](../../docs/EXTENDED_FEATURES.md) - Extended feature details
- [Tests README](../unit/README.md) - Unit test documentation
