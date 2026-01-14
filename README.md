# veripy

<p align="center">
  <strong>A Production-Ready Auto-Active Verification System for Python</strong>
</p>

<p align="center">
  <a href="https://github.com/veripy/veripy">
    <img src="https://img.shields.io/badge/GitHub-veripy-blue" alt="GitHub">
  </a>
  <a href="https://pypi.org/project/veripy/">
    <img src="https://img.shields.io/badge/PyPI-veripy-blue" alt="PyPI">
  </a>
  <a href="https://github.com/veripy/veripy/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
</p>

## Overview

Veripy is a production-ready auto-active verification system for Python programs, inspired by Dafny and Verus. It provides automated theorem proving capabilities using SMT (Satisfiability Modulo Theories) solving, enabling developers to verify that their Python code satisfies specified contracts and properties.

### What is Auto-Active Verification?

Auto-active verification combines the best of both worlds:
- **Automated**: The verifier automatically handles routine proof steps
- **Active**: Users provide high-level specifications (contracts) and hints when needed

This approach significantly reduces the annotation burden while maintaining strong correctness guarantees.

## Features

### Core Verification
- [x] Basic verification over ints/bools (assign/if/while/assert/assume)
- [x] Quantifiers in contracts: `forall`, `exists` (typed as `: int`/`: bool`)
- [x] `Havoc` for `while` encoding and weakest preconditions
- [x] Arrays via theory of arrays: `xs[i]`, `xs[i] = v` (functional `store`)
- [x] Recursive function verification with termination checking
- [x] Function calls and cross-function verification
- [x] Refinement types for precise specifications

### Auto-Active Features
- [x] **Automatic invariant inference**: Infers loop bounds and type constraints
- [x] **Lemma generation**: Automatically generates arithmetic lemmas (commutativity, associativity)
- [x] **Termination checking**: Validates decreases clauses for recursive functions
- [x] **Caching**: Caches verification results for incremental analysis

### Production Features
- [x] **Rich CLI**: Production-grade command-line interface with multiple output formats
- [x] **Comprehensive error reporting**: Detailed error messages with source locations and counterexamples
- [x] **JSON output**: Machine-readable verification reports for CI/CD integration
- [x] **Progress tracking**: TQDM progress bars for verification progress
- [x] **Configuration**: Flexible verification configuration (timeout, solver selection, caching)

### Developer Experience
- [x] **Type annotations**: Full support for Python type hints
- [x] **Rich terminal output**: Beautiful, readable verification reports
- [x] **Suggestions**: Helpful suggestions for fixing verification failures
- [x] **Statistics**: Detailed verification statistics

## Installation

### Basic Installation

```bash
pip install veripy
```

### Development Installation

```bash
git clone https://github.com/veripy/veripy.git
cd veripy
pip install -e ".[dev]"
```

### With All Dependencies

```bash
pip install -e ".[dev,docs,lsp]"
```

## Quick Start

### Basic Verification

```python
import veripy as vp

# Enable verification
vp.enable_verification()

# Define a function with contracts
@vp.verify(requires=['x > 0', 'y > 0'], ensures=['ans == x + y'])
def add_positive(x: int, y: int) -> int:
    return x + y

# Verify all decorated functions
vp.verify_all()
```

### Loop Verification with Invariants

```python
@vp.verify(requires=['n >= 0'], ensures=['ans == n * (n + 1) // 2'])
def summation(n: int) -> int:
    ans = 0
    i = 0
    while i < n:
        vp.invariant('i >= 0')
        vp.invariant('ans == i * (i - 1) // 2')
        ans = ans + i
        i = i + 1
    return ans
```

### Recursive Function Verification

```python
@vp.verify(
    requires=['n >= 0'],
    ensures=['ans == factorial(n)'],
    decreases='n'
)
def factorial(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
```

### Using Quantifiers

```python
@vp.verify(
    requires=[],
    ensures=['forall(i: int :: i == i)']
)
def quantifier_example() -> bool:
    return True
```

## Command Line Interface

Veripy provides a powerful CLI for verification:

### Verify Files

```bash
veripy verify file.py

# With specific functions
veripy verify --function add --function multiply file.py

# With counterexamples
veripy verify --counterexample file.py

# JSON output for CI/CD
veripy verify --output json file.py
```

### Check for Annotations

```bash
veripy check file.py
```

### Get Version Info

```bash
veripy version
veripy info
```

## Auto-Active Features

Veripy can automatically infer many invariants and lemmas:

```python
from veripy.auto_active import (
    auto_infer_invariants,
    generate_arithmetic_lemmas,
    register_lemma
)

# Auto-infer loop invariants
invariants = auto_infer_invariants({
    "loop_var": "i",
    "init": 0,
    "condition": "i < n",
    "body": []
})

# Generate arithmetic lemmas
lemmas = generate_arithmetic_lemmas("x + y")
# Returns: ['forall(x: int, y: int :: x + y == y + x)', ...]

# Register custom lemmas
register_lemma(
    name="my_lemma",
    premises=["x > 0"],
    conclusion="x >= 0"
)
```

## Error Reporting

Veripy provides comprehensive error reporting:

```python
from veripy.error_reporter import (
    report_verification_failure,
    print_verification_report
)

# Report a failure
report_verification_failure(
    message="Precondition not satisfied",
    category=ErrorCategory.PRECONDITION,
    model=solver.model(),
    variables={"x": int, "y": int}
)

# Print detailed report
print_verification_report(output_format="text")
# or
print_verification_report(output_format="json")
```

## Architecture

```
veripy/
├── core/                    # Core verification engine
│   ├── verify.py           # Verification condition generation
│   ├── transformer.py      # AST transformation
│   └── wp.py               # Weakest precondition calculus
├── auto_active/            # Auto-active features
│   ├── invariant_inference.py
│   └── lemma_engine.py
├── cli/                    # Command-line interface
│   └── main.py
├── parser/                 # Specification parsing
│   ├── syntax.py
│   └── parser.py
├── typecheck/             # Type system
│   ├── types.py
│   └── type_check.py
├── error_reporter.py      # Error reporting
└── prettyprint.py         # Output formatting
```

## Requirements

### Binaries
- [Python 3.9+](https://www.python.org/)
- [Z3 SMT Solver](https://github.com/Z3Prover/z3)

### Python Libraries
- z3-solver (SMT solver bindings)
- pyparsing (Expression parsing)
- apronpy (Abstract interpretation)
- rich (Terminal output)
- click (CLI framework)
- tqdm (Progress bars)

## Testing

Run the test suite:

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make test-cov

# Fast mode (stop on first failure)
make test-fast
```

## Development

```bash
# Install development dependencies
make install-dev

# Run linting
make lint

# Run formatters
make format

# Type checking
make typecheck

# All checks
make check
```

## Documentation

Build documentation:

```bash
make docs
```

View live documentation:

```bash
make docs-live
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Roadmap

### Near-term (v0.2.0)
- [ ] Improved invariant inference with machine learning
- [ ] VS Code extension with LSP
- [ ] Performance optimizations
- [ ] Additional SMT solver support (CVC5)

### Mid-term (v0.3.0)
- [ ] Module system with cross-file verification
- [ ] Class/object verification
- [ ] Exception handling verification
- [ ] Property-based testing integration

### Long-term (v1.0.0)
- [ ] Full Python language support
- [ ] Web-based verification IDE
- [ ] Cloud verification service
- [ ] Certification and compliance features

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Z3 Theorem Prover](https://github.com/Z3Prover/z3) for the SMT solving engine
- [Dafny](https://github.com/dafny-lang/dafny) for inspiring the auto-active verification approach
- [Verus](https://github.com/verus-lang/verus) for demonstrating auto-active verification in practice
- [APRON](http://apron.cri.ensmp.fr/) for abstract interpretation domain

---

<p align="center">
  Made with ❤️ by the Veripy Team
</p>