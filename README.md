# Veripy

A production-ready auto-active verification system for Python, similar to Dafny and Verus.

## Overview

Veripy transforms Python code verification from a work-in-progress into a production-ready system with automatic invariant inference, comprehensive error reporting, and a professional CLI.

## Key Features

### Auto-Active Verification
- Automatic loop invariant inference
- Type-based constraint generation
- Arithmetic lemma automation
- Multiple inference strategies (NONE, SIMPLE, AGGRESSIVE, FULL)

### Production Readiness
- Professional CLI with multiple commands
- Rich terminal output with syntax highlighting
- JSON output for CI/CD
- Comprehensive error reporting with counterexamples
- Source location tracking

### Recursive Functions
- Termination checking with decreases clauses
- Tail recursion detection
- Call graph analysis

## Installation

```bash
pip install veripy
```

For development:
```bash
make install-dev
```

## Usage

### Basic Verification
```python
import veripy as vp

vp.enable_verification()

@vp.verify(requires=['x > 0'], ensures=['ans == x * 2'])
def double(x: int) -> int:
    return x * 2

vp.verify_all()
```

### CLI Usage
```bash
# Verify files
veripy verify file.py

# With counterexamples and statistics
veripy verify --counterexample --statistics file.py

# JSON output for CI/CD
veripy verify --output json file.py
```

### Auto-Active Features
```python
from veripy.auto_active import auto_infer_invariants

invariants = auto_infer_invariants({
    "loop_var": "i",
    "init": 0,
    "condition": "i < n",
    "body": []
})
```

## Architecture

```
veripy/
├── core/           # Verification condition generation, AST transformation
├── auto_active/    # Auto-active features (invariant inference, lemmas)
├── cli/            # Command-line interface
├── parser/         # Syntax parsing
├── typecheck/      # Type checking
├── error_reporter.py
├── recursive.py
└── prettyprint.py
```

## Development

```bash
make test          # Run tests
make test-cov      # With coverage
make lint          # Linting
make format        # Formatting
make typecheck     # Type checking
make check         # All checks
```

## Dependencies

- `z3-solver` - SMT solver
- `rich` - Terminal output
- `click` - CLI framework
- `pyparsing` - Expression parsing
- `apronpy` - Abstract interpretation

## Roadmap

- Improved invariant inference with ML
- VS Code extension with LSP
- Performance optimizations
- Module system and class/object verification
