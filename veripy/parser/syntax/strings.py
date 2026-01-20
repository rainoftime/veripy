"""String support classes for veripy syntax."""

from typing import Optional
from .expressions import Expr


class StringLiteral(Expr):
    """String literal."""
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return f'(StringLiteral "{self.value}")'
    
    def variables(self):
        return set()


class StringConcat(Expr):
    """String concatenation: s1 + s2"""
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f'(StringConcat {self.left} + {self.right})'
    
    def variables(self):
        return self.left.variables().union(self.right.variables())


class StringLength(Expr):
    """String length: len(s)"""
    def __init__(self, string_expr: Expr):
        self.string_expr = string_expr
    
    def __repr__(self):
        return f'(StringLength len({self.string_expr}))'
    
    def variables(self):
        return self.string_expr.variables()


class StringIndex(Expr):
    """String indexing: s[i]"""
    def __init__(self, string_expr: Expr, index: Expr):
        self.string_expr = string_expr
        self.index = index
    
    def __repr__(self):
        return f'(StringIndex {self.string_expr}[{self.index}])'
    
    def variables(self):
        return self.string_expr.variables().union(self.index.variables())


class StringSubstring(Expr):
    """String substring: s[i:j]"""
    def __init__(self, string_expr: Expr, start: Expr, end: Optional[Expr] = None):
        self.string_expr = string_expr
        self.start = start
        self.end = end
    
    def __repr__(self):
        end_str = f':{self.end}' if self.end else ''
        return f'(StringSubstring {self.string_expr}[{self.start}{end_str}])'
    
    def variables(self):
        result = self.string_expr.variables().union(self.start.variables())
        if self.end:
            result.update(self.end.variables())
        return result


class StringContains(Expr):
    """Check if substring in string: sub in s"""
    def __init__(self, substring: Expr, string_expr: Expr):
        self.substring = substring
        self.string_expr = string_expr
    
    def __repr__(self):
        return f'(StringContains {self.substring} in {self.string_expr})'
    
    def variables(self):
        return self.substring.variables().union(self.string_expr.variables())
