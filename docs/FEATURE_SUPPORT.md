# Python Feature Support Matrix for Veripy

This document provides a complete overview of which Python features are supported and which are not.

## Feature Support Overview

| Feature | Status | Notes |
|---------|--------|-------|
| **Control Flow** | | |
| Variable assignment | ✅ Supported | `x = expr` |
| If/elif/else | ✅ Supported | Full support |
| While loops | ✅ Supported | With invariants |
| For loops | ⚠️ Partial | Only `range()` iteration |
| Match/case | ❌ Not supported | Python 3.10+ pattern matching |
| Break/continue | ❌ Not supported | Loop control flow |
| Try/except/finally | ❌ Not supported | Exception handling |
| Raise | ❌ Not supported | Exception raising |
| With statement | ❌ Not supported | Context managers |
| Pass | ✅ Supported | No-op statement |
| Assert | ✅ Supported | Verification assertions |
| Return | ✅ Supported | Function return |
| Yield | ❌ Not supported | Generators |
| Yield from | ❌ Not supported | Delegating to subgenerator |
| Async/await | ❌ Not supported | Asynchronous code |
| Await | ❌ Not supported | Awaiting coroutines |
| Async for | ❌ Not supported | Async iteration |
| Async with | ❌ Not supported | Async context managers |
| Global/nonlocal | ❌ Not supported | Scope modifiers |
| Import statements | ❌ Not supported | Module imports |
| From ... import | ❌ Not supported | Selective imports |
| Import as | ❌ Not supported | Aliasing imports |
| **Data Types** | | |
| Integers | ✅ Supported | `int` type |
| Booleans | ✅ Supported | `True`, `False` |
| Strings | ✅ Supported | Basic operations |
| Lists | ⚠️ Partial | Array theory only |
| Tuples | ❌ Not supported | Immutable sequences |
| Sets | ✅ Supported | Union, intersection, etc. |
| Dictionaries | ✅ Supported | Get, set, keys, values |
| Frozen sets | ❌ Not supported | Immutable sets |
| None | ⚠️ Partial | Limited support |
| Bytes | ❌ Not supported | Binary data |
| Bytearray | ❌ Not supported | Mutable bytes |
| Memoryview | ❌ Not supported | Memory buffer views |
| Range | ✅ Supported | In for loops |
| Complex numbers | ❌ Not supported | Complex arithmetic |
| Fractions | ❌ Not supported | Rational numbers |
| Decimals | ❌ Not supported | Decimal arithmetic |
| **Expressions** | | |
| Literals | ✅ Supported | `1`, `True`, `"hello"` |
| Variables | ✅ Supported | Simple names |
| Binary operators | ✅ Supported | `+`, `-`, `*`, `/`, `%`, `//`, `**` |
| Unary operators | ✅ Supported | `-`, `+`, `not` |
| Comparison operators | ✅ Supported | `<`, `<=`, `>`, `>=`, `==`, `!=` |
| Boolean operators | ✅ Supported | `and`, `or`, `not` |
| Bitwise operators | ❌ Not supported | `&`, `|`, `^`, `<<`, `>>` |
| Membership tests | ✅ Supported | `in`, `not in` |
| Identity tests | ❌ Not supported | `is`, `is not` |
| Conditional expr | ❌ Not supported | `x if cond else y` |
| Lambda | ❌ Not supported | Anonymous functions |
| List comprehension | ✅ Supported | `[x for x in iter]` |
| Set comprehension | ✅ Supported | `{x for x in iter}` |
| Dict comprehension | ⚠️ Partial | Limited support |
| Generator expr | ❌ Not supported | `(x for x in iter)` |
| Function calls | ⚠️ Partial | User-defined limited |
| Method calls | ✅ Supported | `obj.method(args)` |
| Subscript | ✅ Supported | `arr[i]`, `arr[i:j:k]` |
| Attribute access | ✅ Supported | `obj.field` |
| String formatting | ❌ Not supported | `f"..."`, `"{}".format()` |
| **Functions** | | |
| Function def | ✅ Supported | `def f(x):` |
| Parameters | ⚠️ Partial | Positional only |
| Default arguments | ⚠️ Partial | Limited |
| Keyword arguments | ❌ Not supported | `f(x=1)` |
| *args | ❌ Not supported | Variadic args |
| **kwargs | ❌ Not supported | Variadic kwargs |
| Decorators | ❌ Not supported | `@decorator` |
| Type hints | ✅ Supported | `: int`, `-> str` |
| Docstrings | ⚠️ Partial | For contracts |
| Nested functions | ❌ Not supported | Function scope |
| Closures | ❌ Not supported | Captured variables |
| Recursion | ✅ Supported | With termination proof |
| **Classes** | | |
| Class def | ⚠️ Partial | Definition only |
| Instance attributes | ⚠️ Partial | Field access |
| Class attributes | ❌ Not supported | Shared state |
| Methods | ⚠️ Partial | Basic support |
| Static method | ❌ Not supported | `@staticmethod` |
| Class method | ❌ Not supported | `@classmethod` |
| Property | ❌ Not supported | `@property` |
| Inheritance | ❌ Not supported | Subclassing |
| Super() | ❌ Not supported | Parent access |
| MRO | ❌ Not supported | Method resolution |
| Abstract classes | ❌ Not supported | ABC |
| Magic methods | ❌ Not supported | `__init__`, `__str__`, etc. |
| Class invariants | ⚠️ Partial | Definition only |
| Constructor | ⚠️ Partial | `__init__` |
| Destructor | ❌ Not supported | `__del__` |
| Instance checks | ❌ Not supported | `isinstance()` |
| **Modules** | | |
| Module creation | ❌ Not supported | `module.py` |
| Package creation | ❌ Not supported | `package/__init__.py` |
| Relative import | ❌ Not supported | `.module` |
| Import from | ❌ Not supported | `from ... import` |
| __all__ | ❌ Not supported | Export list |
| **OOP Features** | | |
| Single inheritance | ❌ Not supported | Class hierarchy |
| Multiple inheritance | ❌ Not supported | Diamond problem |
| Method overriding | ❌ Not supported | Polymorphism |
| Method overloading | ❌ Not supported | Type-based dispatch |
| Abstract methods | ❌ Not supported | Unimplemented methods |
| Interface | ❌ Not supported | Protocol type |
| Mixins | ❌ Not supported | Code reuse |
| Slots | ❌ Not supported | `__slots__` |
| Descriptors | ❌ Not supported | `__get__`, `__set__` |
| **Standard Library** | | |
| math | ❌ Not supported | Mathematical functions |
| collections | ❌ Not supported | Container types |
| itertools | ❌ Not supported | Iterator tools |
| functools | ❌ Not supported | Higher-order functions |
| operator | ❌ Not supported | Standard operators |
| typing | ⚠️ Partial | Basic type hints |
| dataclasses | ❌ Not supported | Auto `__init__` |
| enum | ❌ Not supported | Enumeration types |
| random | ❌ Not supported | Random numbers |
| datetime | ❌ Not supported | Date/time types |
| json | ❌ Not supported | JSON serialization |
| re | ❌ Not supported | Regular expressions |
| **Concurrency** | | |
| Threads | ❌ Not supported | `threading` |
| Locks | ❌ Not supported | Synchronization |
| Queues | ❌ Not supported | Thread-safe queues |
| Processes | ❌ Not supported | `multiprocessing` |
| AsyncIO | ❌ Not supported | Event loops |
| Coroutines | ❌ Not supported | `async def` |
| Futures | ❌ Not supported | `concurrent.futures` |
| **Testing** | | |
| assert | ✅ Supported | Assertion statements |
| unittest | ❌ Not supported | Test framework |
| pytest | ❌ Not supported | Test framework |
| Hypothesis | ❌ Not supported | Property-based testing |

## Detailed Unsupported Features

### 1. Exception Handling
```python
# NOT SUPPORTED
try:
    risky_operation()
except ValueError as e:
    handle_error(e)
except Exception:
    log("error")
finally:
    cleanup()
```

**Why**: Veripy uses Hoare logic which doesn't naturally model exception control flow. Exception handling requires different verification approaches.

### 2. Context Managers
```python
# NOT SUPPORTED
with open("file.txt") as f:
    data = f.read()
```

**Why**: Context managers require modeling of resource acquire/release patterns and rely on exception handling internally.

### 3. Generators and Yield
```python
# NOT SUPPORTED
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

**Why**: Generators involve complex state machines and partial function termination, significantly complicating verification.

### 4. Classes and Inheritance
```python
# NOT SUPPORTED
class Animal:
    def speak(self):
        raise NotImplementedError()

class Dog(Animal):
    def speak(self):
        return "woof"
```

**Why**: Full OOP verification requires:
- Inheritance hierarchies
- Method override checking
- Liskov substitution principle
- Frame conditions for `self`
- Invariant inheritance

### 5. Async/Await
```python
# NOT SUPPORTED
async def fetch_data():
    response = await http.get(url)
    return response.json()
```

**Why**: Async verification requires modeling:
- Event loops
- Task scheduling
- Concurrency invariants
- Promise/future semantics

### 6. Closures
```python
# NOT SUPPORTED
def make_adder(n):
    def adder(x):
        return x + n  # n is captured
    return adder
```

**Why**: Closure verification requires:
- Captured variable tracking
- Environment modeling
- Higher-order function verification

### 7. Bitwise Operations
```python
# NOT SUPPORTED
result = x & 0xFF | (y << 8)
```

**Why**: Bit-level operations require:
- Bit-vector theory
- Endianness considerations
- Hardware-specific behaviors

### 8. String Formatting
```python
# NOT SUPPORTED
msg = f"Hello {name}, you have {count} messages"
```

**Why**: String formatting is complex and requires modeling of format specifications.

## Features Under Development

These features are planned but not yet fully implemented:

| Feature | Status | ETA |
|---------|--------|-----|
| Class verification | In progress | v0.3 |
| Exception handling | Planned | v0.4 |
| Generators | Researching | Future |
| String patterns | In progress | v0.3 |

## Workarounds

For unsupported features, consider these approaches:

### Exception Handling
```python
# Instead of try/except, use preconditions
@verify(requires=['x != 0'])
def divide(x, y):
    return y / x  # x != 0 guaranteed by precondition
```

### Context Managers
```python
# Instead of with, use explicit acquire/release
@verify(requires=['resource is available'])
def use_resource(resource):
    do_work(resource)
    resource.release()
```

### Generators
```python
# Instead of generators, use explicit state
@verify(requires=['state >= 0'])
def next_fib(state):
    # Manual state machine
    return state, state + 1
```

### Classes
```python
# Instead of classes, use records with functions
Point = tuple[int, int]

@verify(ensures=['result[0] >= 0', 'result[1] >= 0'])
def create_point(x, y):
    return (abs(x), abs(y))

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
```

## Feature Requests

To request support for a specific feature, please open an issue at:
https://github.com/veripy/veripy/issues

Include:
1. Use case description
2. Why verification is important for this feature
3. Any relevant research papers or implementations
