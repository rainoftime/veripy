"""Core expression classes for veripy syntax."""

from typing import List
from .operations import Op


class Expr:
    """Base class for all expression AST nodes.
    
    We provide structural equality so unit tests can compare freshly
    constructed nodes (e.g. Literal(VInt(5))) by value rather than identity.
    """
    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        # Most nodes are simple Python objects storing fields in __dict__.
        # (Value uses __slots__ and is handled separately above.)
        return getattr(self, '__dict__', None) == getattr(other, '__dict__', None)
    
    def __ne__(self, other):
        return not self.__eq__(other)


class Var(Expr):
    """Variable expression."""
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f'(Var {self.name})'
    
    def variables(self):
        return {self.name}


class Literal(Expr):
    """Literal expression."""
    def __init__(self, v):
        self.value = v
    
    def __repr__(self):
        return f'(Literal {self.value})'
    
    def variables(self):
        return set()


class BinOp(Expr):
    """Binary operation expression."""
    def __init__(self, l: Expr, op: Op, r: Expr):
        self.e1 = l
        self.e2 = r
        self.op = op
    
    def __repr__(self):
        return f'(BinOp {self.e1} {self.op} {self.e2})'
    
    def variables(self):
        return {*self.e1.variables(), *self.e2.variables()}


class UnOp(Expr):
    """Unary operation expression."""
    def __init__(self, op: Op, expr: Expr):
        self.op = op
        self.e = expr
    
    def __repr__(self):
        return f'(UnOp {self.op} {self.e})'
    
    def variables(self):
        return {*self.e.variables()}


class Quantification(Expr):
    """Quantification expression (forall)."""
    def __init__(self, var: Var, ty, expr: Expr):
        self.var = var
        self.ty = ty
        self.expr = expr
    
    def __repr__(self):
        return f'(∀{self.var} : {self.ty}. {self.expr})'
    
    def variables(self):
        return {self.var.name, *self.expr.variables()}


class Subscript(Expr):
    """Array/list subscript expression."""
    def __init__(self, var, subscript):
        self.var = var
        self.subscript = subscript
    
    def __repr__(self):
        return f'(Subscript {self.var} [{self.subscript}])'
    
    def variables(self):
        return self.var.variables().union(self.subscript.variables())


class Slice(Expr):
    """Represents a slice expression: a[lower:upper:step]"""
    def __init__(self, lower=None, upper=None, step=None):
        self.lower = lower
        self.upper = upper
        self.step = step
    
    def __repr__(self):
        lower_str = str(self.lower) if self.lower else ''
        upper_str = str(self.upper) if self.upper else ''
        step_str = f':{self.step}' if self.step else ''
        return f'(Slice {lower_str}:{upper_str}{step_str})'
    
    def variables(self):
        result = set()
        if self.lower and isinstance(self.lower, Expr):
            result.update(self.lower.variables())
        if self.upper and isinstance(self.upper, Expr):
            result.update(self.upper.variables())
        if self.step and isinstance(self.step, Expr):
            result.update(self.step.variables())
        return result


class Store(Expr):
    """Array store expression."""
    def __init__(self, arr, idx, val):
        self.arr = arr
        self.idx = idx
        self.val = val
    
    def __repr__(self):
        return f'(Store {self.arr} [{self.idx}] = {self.val})'
    
    def variables(self):
        return self.arr.variables().union(self.idx.variables()).union(self.val.variables())


class FunctionCall(Expr):
    """Function call expression."""
    def __init__(self, func_name, args, native=True):
        self.func_name = func_name
        self.args = args
        self.native = native
    
    def __repr__(self):
        return f'(Call {self.func_name} with ({self.args}))'
    
    def variables(self):
        result = set()
        for arg in self.args:
            if isinstance(arg, Expr):
                result.update(arg.variables())
        return result


class Old(Expr):
    """Represents old(expr) for referring to pre-state values in postconditions"""
    def __init__(self, expr: Expr):
        self.expr = expr
    
    def __repr__(self):
        return f'(Old {self.expr})'
    
    def variables(self):
        return self.expr.variables()


class RecordField(Expr):
    """Represents a record/struct field access"""
    def __init__(self, record: Expr, field: str):
        self.record = record
        self.field = field
    
    def __repr__(self):
        return f'(RecordField {self.record}.{self.field})'
    
    def variables(self):
        return self.record.variables()
