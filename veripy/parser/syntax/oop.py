"""OOP (Object-Oriented Programming) support classes for veripy syntax."""

from typing import List, TYPE_CHECKING
from .expressions import Expr, Var

if TYPE_CHECKING:
    from veripy.typecheck.types import Type


class FieldAccess(Expr):
    """Access to an object's field: obj.field"""
    def __init__(self, obj: Expr, field: str):
        self.obj = obj
        self.field = field
    
    def __repr__(self):
        return f'(FieldAccess {self.obj}.{self.field})'
    
    def variables(self):
        return self.obj.variables()


class FieldAssign(Expr):
    """Assignment to an object's field: obj.field = value"""
    def __init__(self, obj: Expr, field: str, value: Expr):
        self.obj = obj
        self.field = field
        self.value = value
    
    def __repr__(self):
        return f'(FieldAssign {self.obj}.{self.field} = {self.value})'
    
    def variables(self):
        return self.obj.variables().union(self.value.variables())


class ClassInvariant:
    """Class invariant assertion."""
    def __init__(self, class_name: str, expr: Expr):
        self.class_name = class_name
        self.expr = expr
    
    def __repr__(self):
        return f'(Invariant {self.class_name} :: {self.expr})'
    
    def variables(self):
        return self.expr.variables()


class MethodCall(Expr):
    """Method call on an object: obj.method(args)"""
    def __init__(self, obj: Expr, method_name: str, args: List[Expr]):
        self.obj = obj
        self.method_name = method_name
        self.args = args
    
    def __repr__(self):
        return f'(MethodCall {self.obj}.{self.method_name}({self.args}))'
    
    def variables(self):
        result = self.obj.variables()
        for arg in self.args:
            if isinstance(arg, Expr):
                result.update(arg.variables())
        return result


class ClassDef:
    """Class definition for verification."""
    def __init__(self, name: str, fields: List[tuple], 
                 invariants: List[Expr] = None,
                 methods: List['MethodDef'] = None):
        self.name = name
        self.fields = fields  # List of (field_name, type)
        self.invariants = invariants or []
        self.methods = methods or []
    
    def __repr__(self):
        return f'(ClassDef {self.name} fields={self.fields} invariants={self.invariants})'


class MethodDef:
    """Method definition for verification."""
    def __init__(self, class_name: str, name: str, 
                 params: List[tuple], 
                 requires: List[Expr] = None,
                 ensures: List[Expr] = None,
                 body = None):
        self.class_name = class_name
        self.name = name
        self.params = params  # List of (param_name, type)
        self.requires = requires or []
        self.ensures = ensures or []
        self.body = body
    
    def __repr__(self):
        return f'(MethodDef {self.class_name}.{self.name}({self.params}))'
