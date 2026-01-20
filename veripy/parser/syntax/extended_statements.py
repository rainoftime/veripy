"""Extended statement types for veripy syntax."""

from typing import List
from .statements import Stmt
from .expressions import Expr
from .operations import ArithOps


class Try(Stmt):
    """Try/except/finally statement for exception handling."""
    def __init__(self, body: Stmt, handlers: List['ExceptHandler'], 
                 orelse: Stmt = None, finalbody: Stmt = None):
        self.body = body
        self.handlers = handlers  # List of (exception_type, var_name, handler_body)
        self.orelse = orelse  # else clause
        self.finalbody = finalbody  # finally clause
    
    def __repr__(self):
        handlers_str = ', '.join(str(h) for h in self.handlers)
        return f'(Try body={self.body} handlers=[{handlers_str}] orelse={self.orelse} finalbody={self.finalbody})'
    
    def variables(self):
        result = self.body.variables()
        for handler in self.handlers:
            result.update(handler.body.variables())
        if self.orelse:
            result.update(self.orelse.variables())
        if self.finalbody:
            result.update(self.finalbody.variables())
        return result


class ExceptHandler:
    """Exception handler in try statement."""
    def __init__(self, exc_type: str, exc_var: str = None, body: Stmt = None):
        self.exc_type = exc_type  # Exception type name
        self.exc_var = exc_var  # Optional exception variable name
        self.body = body  # Handler body
    
    def __repr__(self):
        var_str = f' as {self.exc_var}' if self.exc_var else ''
        return f'(ExceptHandler {self.exc_type}{var_str} body={self.body})'


class With(Stmt):
    """With statement for context management."""
    def __init__(self, context_expr: Expr, item_var: str, body: Stmt):
        self.context_expr = context_expr
        self.item_var = item_var  # Variable bound to context manager result
        self.body = body
    
    def __repr__(self):
        return f'(With {self.item_var} = {self.context_expr} body={self.body})'
    
    def variables(self):
        return self.body.variables()


class ForLoop(Stmt):
    """For loop statement."""
    def __init__(self, loop_var: str, iterable: Expr, body: Stmt, 
                 orelse: Stmt = None):
        self.loop_var = loop_var
        self.iterable = iterable
        self.body = body
        self.orelse = orelse  # Optional else clause
    
    def __repr__(self):
        return f'(For {self.loop_var} in {self.iterable} body={self.body} orelse={self.orelse})'
    
    def variables(self):
        result = self.body.variables()
        if self.orelse:
            result.update(self.orelse.variables())
        result.add(self.loop_var)
        return result


class AugAssign(Stmt):
    """Augmented assignment (e.g., x += 1)."""
    def __init__(self, var: Expr, op: ArithOps, expr: Expr):
        self.var = var
        self.op = op
        self.expr = expr
    
    def __repr__(self):
        return f'(AugAssign {self.var} {self.op}= {self.expr})'
    
    def variables(self):
        return self.var.variables().union(self.expr.variables())


class Break(Stmt):
    """Break statement for loop termination."""
    def __repr__(self):
        return '(Break)'
    
    def variables(self):
        return set()


class Continue(Stmt):
    """Continue statement for loop iteration."""
    def __repr__(self):
        return '(Continue)'
    
    def variables(self):
        return set()


class Raise(Stmt):
    """Raise statement for exception throwing."""
    def __init__(self, exc_expr: Expr = None, cause: Expr = None):
        self.exc_expr = exc_expr  # Exception to raise
        self.cause = cause  # Optional cause
    
    def __repr__(self):
        if self.cause:
            return f'(Raise {self.exc_expr} from {self.cause})'
        return f'(Raise {self.exc_expr})'
    
    def variables(self):
        result = set()
        if self.exc_expr:
            result.update(self.exc_expr.variables())
        if self.cause:
            result.update(self.cause.variables())
        return result


class Global(Stmt):
    """Global variable declaration."""
    def __init__(self, names: List[str]):
        self.names = names
    
    def __repr__(self):
        return f'(Global {self.names})'
    
    def variables(self):
        return set(self.names)


class Nonlocal(Stmt):
    """Nonlocal variable declaration."""
    def __init__(self, names: List[str]):
        self.names = names
    
    def __repr__(self):
        return f'(Nonlocal {self.names})'
    
    def variables(self):
        return set(self.names)


class ImportStmt(Stmt):
    """Import statement."""
    def __init__(self, module: str, names: List[str] = None, alias: str = None):
        self.module = module  # Module to import
        self.names = names or []  # Specific names to import (None means import *)
        self.alias = alias  # Optional alias for the module
    
    def __repr__(self):
        if self.alias:
            return f'(Import {self.module} as {self.alias})'
        if self.names:
            return f'(Import {self.module}.{{{", ".join(self.names)}}})'
        return f'(Import {self.module})'
    
    def variables(self):
        return set()


class ImportFrom(Stmt):
    """From...import statement."""
    def __init__(self, module: str, names: List[str], alias: List[str] = None):
        self.module = module  # Module to import from
        self.names = names  # Names to import
        self.alias = alias or []  # Optional aliases
    
    def __repr__(self):
        result = f'(From {self.module} Import '
        if self.alias:
            pairs = [f'{n} as {a}' for n, a in zip(self.names, self.alias)]
            result += ', '.join(pairs)
        else:
            result += ', '.join(self.names)
        return result + ')'
    
    def variables(self):
        return set()
