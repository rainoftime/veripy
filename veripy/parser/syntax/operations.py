"""Operation enumerations for veripy syntax."""

from enum import Enum


class Op:
    """Base class for all operations."""
    pass


class ArithOps(Op, Enum):
    """Arithmetic operations."""
    Add = '+'
    Minus = '-'
    Mult = '*'
    IntDiv = '/'
    Neg = '-'
    Mod = '%'


class CompOps(Op, Enum):
    """Comparison operations."""
    Eq = '='
    Neq = '!='
    Lt = '<'
    Le = '<='
    Gt = '>'
    Ge = '>='
    In = 'in'
    NotIn = 'not in'


class BoolOps(Op, Enum):
    """Boolean operations."""
    And = 'and'
    Or = 'or'
    Not = 'not'
    Implies = '==>'
    Iff = '<==>'


class SetOps(Op, Enum):
    """Set operations."""
    Union = 'union'
    Intersection = 'intersection'
    Difference = 'difference'
    Member = 'in'
    Subset = 'subset'
    Superset = 'superset'


class DictOps(Op, Enum):
    """Dictionary/map operations."""
    Get = 'get'
    Keys = 'keys'
    Values = 'values'
    Contains = 'contains'
