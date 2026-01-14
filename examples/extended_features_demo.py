"""
Example programs demonstrating extended Python features in veripy.

This file shows how to use:
- Classes/OOP with verification
- Set operations (union, intersection, membership, cardinality)
- Dictionary operations (get, set, keys, values)
- String operations (concat, length, substring, index)
- List comprehensions
"""

# ============================================================================
# Example 1: Classes with Verification
# ============================================================================

"""
Python example (conceptual - actual veripy syntax may vary):

class Counter:
    def __init__(self):
        self.value = 0
        self.min_value = 0
    
    def increment(self):
        self.value = self.value + 1
        return self.value
    
    def decrement(self):
        self.value = self.value - 1
        return self.value
    
    # Class invariant: value >= min_value
    # Method postcondition: value >= min_value

# Usage:
counter = Counter()
counter.increment()
counter.increment()
assert counter.value == 2
"""

# ============================================================================
# Example 2: Set Operations
# ============================================================================

"""
Veripy set operations:

# Set literals
s1 = {1, 2, 3}
s2 = {2, 3, 4}

# Set operations
union = s1 union s2           # {1, 2, 3, 4}
intersection = s1 intersect s2  # {2, 3}
difference = s1 - s2           # {1}
membership = 2 in s1           # True
cardinality = |s1|             # 3
subset = s1 subset s2          # False

# Set membership verification
@verify(requires=['x in s'], ensures=['x >= 0'])
def check_positive(x, s):
    return True

# Cardinality verification
@verify(requires=['|s| > 0'], ensures=['s != empty'])
def non_empty_set(s):
    return True
"""

# ============================================================================
# Example 3: Dictionary Operations
# ============================================================================

"""
Veripy dictionary operations:

# Dict literals
d = {'a': 1, 'b': 2, 'c': 3}

# Dict operations
value = d['a']                    # Get: 1
new_d = d['d'] = 4                # Set: creates new dict
keys = keys(d)                    # Keys: {'a', 'b', 'c'}
contains = 'a' in d               # Contains: True
default = get(d, 'x', 0)          # Get with default: 0

# Dict verification
@verify(requires=['k in d'], ensures=['get(d, k) >= 0'])
def get_positive(d, k):
    return True
"""

# ============================================================================
# Example 4: String Operations
# ============================================================================

"""
Veripy string operations:

# String literals
s1 = "hello"
s2 = "world"

# String operations
concat = s1 + s2                  # "helloworld"
length = len(s1)                  # 5
index = s1[0]                     # 'h'
substring = s1[0:2]               # 'he'
contains = "ell" in s1            # True

# String verification
@verify(requires=['len(s) > 0'], ensures=['s[0] == s[0]'])
def first_char(s):
    return True
"""

# ============================================================================
# Example 5: List Comprehensions
# ============================================================================

"""
Veripy list comprehensions:

# List comprehension
squares = [x * x for x in range(10) if x > 0]

# Set comprehension
evens = {x for x in range(10) if x % 2 == 0}

# Dict comprehension
squares_dict = {x: x * x for x in range(5)}

# Comprehension verification
@verify(requires=['n > 0'], ensures=['len([x for x in range(n)]) == n'])
def list_length(n):
    return True
"""

# ============================================================================
# Example 6: Combined Operations
# ============================================================================

"""
A more complex example combining multiple features:

class BankAccount:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.min_balance = 0
        self.transactions = {}
    
    def deposit(self, amount):
        self.balance = self.balance + amount
        self.transactions[get_time()] = amount
        return self.balance
    
    def withdraw(self, amount):
        assert amount <= self.balance
        self.balance = self.balance - amount
        return self.balance
    
    # Invariant: balance >= min_balance
    # Invariant: balance >= 0
"""

# ============================================================================
# Z3 Models for Extended Types
# ============================================================================

"""
How veripy translates extended types to Z3:

1. Sets:
   - Represented as Z3 arrays: Array(Int, Bool)
   - union: z3.Function('set_union', Array(Int, Bool), Array(Int, Bool), Array(Int, Bool))
   - intersection: z3.Function('set_inter', ...)
   - cardinality: z3.Function('card_set_int', Array(Int, Bool), Int)
   - membership: z3.Select(set_array, element)

2. Dictionaries:
   - Represented as Z3 arrays: Array(KeyType, ValueType)
   - keys: z3.Function('dom_map', Array(K, V), Array(K, Bool))
   - get: z3.Select(dict_array, key)
   - set: z3.Store(dict_array, key, value)

3. Strings:
   - Use Z3's native String sort
   - concat: z3.Function('str_concat', String, String, String)
   - length: z3.Function('str_len', String, Int)
   - substring: z3.Function('str_substr', String, Int, Int, String)

4. Classes/Objects:
   - Modeled as uninterpreted functions
   - Field access: FieldClass(obj, field_name)
   - Method calls: MethodClass(obj, args)
   - Invariants checked at method entry/exit
"""

print(__doc__)
print("\n" + "=" * 60)
print("Extended Python features are now available in veripy!")
print("=" * 60)
