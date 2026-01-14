# Extended Python Features in Veripy

This document describes the extended Python features added to veripy for production-ready verification.

## Overview

Veripy now supports the following additional Python features:
- Classes/OOP (field access, method calls, class invariants)
- Sets (union, intersection, membership, cardinality)
- Dictionaries (get, set, keys, values, contains)
- Strings (concat, length, substring, index)
- List/set/dict comprehensions

## 1. Classes and OOP Support

### Field Access
```python
# Python syntax (conceptual)
obj.field
obj.x = 5

# Veripy AST
FieldAccess(obj, "field")
FieldAssignStmt(obj, "field", value)
```

### Method Calls
```python
# Python syntax
obj.method(arg1, arg2)

# Veripy AST
MethodCall(obj, "method", [arg1, arg2])
```

### Class Invariants
```python
# Class invariants are checked at:
# 1. Constructor completion
# 2. Method entry
# 3. Method exit

class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value = self.value + 1
    
    # Invariant: value >= 0
    # Postcondition: value >= 0
```

## 2. Set Operations

### Set Literals
```python
# Set literal
{1, 2, 3}

# Veripy AST
SetLiteral([Literal(VInt(1)), Literal(VInt(2)), Literal(VInt(3))])
```

### Set Operations
| Operation | Syntax | AST |
|-----------|--------|-----|
| Union | `s1 union s2` | `SetOp(s1, SetOps.Union, s2)` |
| Intersection | `s1 intersect s2` | `SetOp(s1, SetOps.Intersection, s2)` |
| Difference | `s1 - s2` | `SetOp(s1, SetOps.Difference, s2)` |
| Membership | `x in s` | `SetOp(x, SetOps.Member, s)` |
| Subset | `s1 subset s2` | `SetOp(s1, SetOps.Subset, s2)` |
| Cardinality | `\|s\|` | `SetCardinality(s)` |

### Built-in Functions
| Function | Description | Z3 Translation |
|----------|-------------|----------------|
| `set()` | Empty set | `K(IntSort(), BoolVal(False))` |
| `mem(x, s)` | Membership test | `Select(s, x)` |
| `card(s)` | Cardinality | `card_set_fun(s)` |

### Example Verification
```python
@verify(requires=['x in s'], ensures=['x >= 0'])
def check_positive(x, s):
    return True
```

## 3. Dictionary Operations

### Dict Literals
```python
# Dict literal
{'a': 1, 'b': 2}

# Veripy AST
DictLiteral([StringLiteral("a"), StringLiteral("b")], 
            [Literal(VInt(1)), Literal(VInt(2))])
```

### Dict Operations
| Operation | Syntax | AST |
|-----------|--------|-----|
| Get | `d[key]` | `DictGet(d, key)` |
| Get with default | `get(d, key, default)` | `DictGet(d, key, default)` |
| Set | `d[key] = value` | `DictSet(d, key, value)` |
| Keys | `keys(d)` | `DictKeys(d)` |
| Values | `values(d)` | `DictValues(d)` |
| Contains | `key in d` | `DictContains(d, key)` |

### Built-in Functions
| Function | Description | Z3 Translation |
|----------|-------------|----------------|
| `dict()` | Empty dict | `K(KeySort(), DefaultVal)` |
| `keys(d)` | Key set | `dom_map_fun(d)` |
| `get(d, k, default)` | Get with default | `Select(d, k)` |

### Example Verification
```python
@verify(requires=['k in d', 'get(d, k) >= 0'])
def check_value(d, k):
    return True
```

## 4. String Operations

### String Literals
```python
# String literal
"hello world"

# Veripy AST
StringLiteral("hello world")
```

### String Operations
| Operation | Syntax | AST |
|-----------|--------|-----|
| Concat | `s1 + s2` | `StringConcat(s1, s2)` |
| Length | `len(s)` | `StringLength(s)` |
| Index | `s[i]` | `StringIndex(s, i)` |
| Substring | `s[i:j]` | `StringSubstring(s, i, j)` |
| Contains | `sub in s` | `StringContains(sub, s)` |

### Z3 Translation
| Operation | Z3 Function |
|-----------|-------------|
| Concat | `str_concat(s1, s2)` |
| Length | `str_len(s)` |
| Substring | `str_substr(s, start, end)` |
| Index | `str_index(s, i)` |
| Contains | `Contains(s, sub)` |

### Example Verification
```python
@verify(requires=['len(s) > 0'], ensures=['s[0] == s[0]'])
def first_char(s):
    return True
```

## 5. List/Set/Dict Comprehensions

### List Comprehension
```python
# List comprehension
[x * x for x in range(10) if x > 0]

# Veripy AST
ListComprehension(
    element_expr=Var("x") * Var("x"),
    element_var=Var("x"),
    iterable=FunctionCall(Var("range"), [Literal(VInt(10))]),
    predicate=BinOp(Var("x"), CompOps.Gt, Literal(VInt(0)))
)
```

### Set Comprehension
```python
# Set comprehension
{x for x in range(10) if x % 2 == 0}

# Veripy AST
SetComprehension(
    element_var=Var("x"),
    source=FunctionCall(Var("range"), [Literal(VInt(10))]),
    predicate=BinOp(
        BinOp(Var("x"), ArithOps.Mod, Literal(VInt(2))),
        CompOps.Eq,
        Literal(VInt(0))
    )
)
```

### Dict Comprehension
```python
# Dict comprehension
{x: x * x for x in range(5)}

# Veripy AST (simplified)
DictLiteral([...], [...])
```

## Z3 Type Mappings

| Python Type | Z3 Sort | Notes |
|-------------|---------|-------|
| `int` | `IntSort()` | |
| `bool` | `BoolSort()` | |
| `str` | `StringSort()` | Z3 native string |
| `set[int]` | `Array(Int, Bool)` | Set as array of bools |
| `dict[K, V]` | `Array(K, V)` | Dict as array |
| `list[T]` | `Array(Int, T)` | List as array |

## Uninterpreted Functions

Veripy uses uninterpreted functions for complex operations:

```python
# Set operations
set_union: (Array(Int, Bool), Array(Int, Bool)) -> Array(Int, Bool)
set_inter: (Array(Int, Bool), Array(Int, Bool)) -> Array(Int, Bool)
set_diff: (Array(Int, Bool), Array(Int, Bool)) -> Array(Int, Bool)
card_set_int: (Array(Int, Bool)) -> Int

# Dict operations  
dom_map_int: (Array(Int, Int)) -> Array(Int, Bool)

# String operations
str_concat: (String, String) -> String
str_len: (String) -> Int
str_substr: (String, Int, Int) -> String
str_index: (String, Int) -> Int
```

## Limitations and Future Work

### Current Limitations
1. **Class verification**: Full class verification requires more work on object models
2. **Dict comprehensions**: Simplified to dict literals
3. **String operations**: Substring and index are uninterpreted
4. **List comprehensions**: Simplified to uninterpreted arrays

### Future Enhancements
1. **Full class models**: Proper object invariants and frame conditions
2. **Inheritance**: Support for subclass verification
3. **Method overriding**: Override checking
4. **String patterns**: Regex and pattern matching
5. **Collection operations**: map, filter, reduce

## Testing

Run the extended feature tests:
```bash
python -m pytest tests/unit/test_extended_features.py -v
```

## Examples

See `examples/extended_features_demo.py` for detailed examples of all new features.
