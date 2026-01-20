"""Async/await support classes for veripy syntax."""

from typing import List, TYPE_CHECKING
from .statements import Stmt
from .expressions import Expr

if TYPE_CHECKING:
    from veripy.typecheck.types import Type


class Await(Expr):
    """Await expression for async/await."""
    def __init__(self, value: Expr):
        self.value = value
    
    def __repr__(self):
        return f'(Await {self.value})'
    
    def variables(self):
        return self.value.variables()


class AsyncFunctionDef(Stmt):
    """Async function definition."""
    def __init__(self, name: str, params: List[tuple], 
                 requires: List[Expr] = None,
                 ensures: List[Expr] = None,
                 body: Stmt = None,
                 returns: 'Type' = None):
        self.name = name
        self.params = params
        self.requires = requires or []
        self.ensures = ensures or []
        self.body = body
        self.returns = returns
    
    def __repr__(self):
        return f'(AsyncFunctionDef {self.name}({self.params}) body={self.body})'
    
    def variables(self):
        return self.body.variables() if self.body else set()


class AsyncFor(Stmt):
    """Async for loop."""
    def __init__(self, loop_var: str, iterable: Expr, body: Stmt, 
                 orelse: Stmt = None):
        self.loop_var = loop_var
        self.iterable = iterable
        self.body = body
        self.orelse = orelse
    
    def __repr__(self):
        return f'(AsyncFor {self.loop_var} in {self.iterable} body={self.body})'
    
    def variables(self):
        result = self.body.variables()
        if self.orelse:
            result.update(self.orelse.variables())
        result.add(self.loop_var)
        return result


class AsyncWith(Stmt):
    """Async with statement."""
    def __init__(self, context_expr: Expr, item_var: str, body: Stmt):
        self.context_expr = context_expr
        self.item_var = item_var
        self.body = body
    
    def __repr__(self):
        return f'(AsyncWith {self.item_var} = {self.context_expr} body={self.body})'
    
    def variables(self):
        return self.body.variables()
