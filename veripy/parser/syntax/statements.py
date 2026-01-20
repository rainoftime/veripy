"""Core statement classes for veripy syntax."""

from typing import List
from .expressions import Expr, Var


class Stmt:
    """Base class for all statement AST nodes (structural equality)."""
    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return getattr(self, '__dict__', None) == getattr(other, '__dict__', None)
    
    def __ne__(self, other):
        return not self.__eq__(other)


class Skip(Stmt):
    """Skip statement (no-op)."""
    def __repr__(self):
        return '(Skip)'
    
    def variables(self):
        return set()


class Assign(Stmt):
    """Assignment statement."""
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr
    
    def __repr__(self):
        return f'(Assign {self.var} = {self.expr})'
    
    def variables(self):
        v = set(self.expr.variables())
        if isinstance(self.var, str):
            v.add(self.var)
        elif isinstance(self.var, Var):
            v.add(self.var.name)
        return v


class FieldAssignStmt(Stmt):
    """Assignment to object field."""
    def __init__(self, obj: Expr, field: str, value: Expr):
        self.obj = obj
        self.field = field
        self.value = value
    
    def __repr__(self):
        return f'(FieldAssign {self.obj}.{self.field} = {self.value})'
    
    def variables(self):
        return self.obj.variables().union(self.value.variables())


class If(Stmt):
    """Conditional statement."""
    def __init__(self, cond_expr: Expr, lb_stmt: Stmt, rb_stmt: Stmt = None):
        self.cond = cond_expr
        self.lb = lb_stmt if lb_stmt is not None else Skip()
        self.rb = rb_stmt if rb_stmt is not None else Skip()
    
    def __repr__(self):
        return f'(If {self.cond} then {self.lb} else {self.rb})'
    
    def variables(self):
        return {*self.cond.variables(), *self.lb.variables(), *self.rb.variables()}


class Seq(Stmt):
    """Sequence statement (statement composition)."""
    def __init__(self, s1: Stmt, s2: Stmt):
        self.s1 = s1 if s1 is not None else Skip()
        self.s2 = s2 if s2 is not None else Skip()
    
    def __repr__(self):
        return f'(Seq {self.s1}; {self.s2})'
    
    def variables(self):
        return {*self.s1.variables(), *self.s2.variables()}


class Assume(Stmt):
    """Assume statement."""
    def __init__(self, e: Expr):
        self.e = e
    
    def __repr__(self):
        return f'(Assume {self.e})'
    
    def variables(self):
        return {*self.e.variables()}


class Assert(Stmt):
    """Assert statement."""
    def __init__(self, e: Expr):
        self.e = e
    
    def __repr__(self):
        return f'(Assert {self.e})'
    
    def variables(self):
        return {*self.e.variables()}


class While(Stmt):
    """While loop statement."""
    def __init__(self, invs: List[Expr], cond: Expr, body: Stmt):
        self.cond = cond
        self.invariants = invs
        self.body = body if body is not None else Skip()
    
    def __repr__(self):
        return f'(While {self.cond} {{invs: {self.invariants}}} {self.body})'
    
    def variables(self):
        return {*self.body.variables()}


class Havoc(Stmt):
    """Havoc statement (non-deterministic assignment)."""
    def __init__(self, var):
        self.var = var
    
    def __repr__(self):
        return f'(Havoc {self.var})'
    
    def variables(self):
        return set()
