"""Type system and advanced feature classes for veripy syntax."""

from typing import List, TYPE_CHECKING
from .expressions import Expr

if TYPE_CHECKING:
    from veripy.typecheck.types import Type


class Property(Expr):
    """Property decorator expression for getter/setter."""
    def __init__(self, name: str, getter: bool = True, setter: bool = False):
        self.name = name
        self.is_getter = getter
        self.is_setter = setter
    
    def __repr__(self):
        if self.is_setter:
            return f'(Property {self.name}.setter)'
        return f'(Property {self.name})'
    
    def variables(self):
        return set()


class StaticMethod(Expr):
    """Static method decorator."""
    def __init__(self, func_name: str):
        self.func_name = func_name
    
    def __repr__(self):
        return f'(StaticMethod {self.func_name})'
    
    def variables(self):
        return set()


class ClassMethod(Expr):
    """Class method decorator."""
    def __init__(self, func_name: str):
        self.func_name = func_name
    
    def __repr__(self):
        return f'(ClassMethod {self.func_name})'
    
    def variables(self):
        return set()


class VarArgs(Expr):
    """Variable positional arguments (*args)."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f'(VarArgs {self.name})'
    
    def variables(self):
        return {self.name}


class KwArgs(Expr):
    """Variable keyword arguments (**kwargs)."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f'(KwArgs {self.name})'
    
    def variables(self):
        return {self.name}


class FString(Expr):
    """F-string expression (Python 3.6+)."""
    def __init__(self, parts: List[Expr], literal_parts: List[str]):
        self.parts = parts  # Expressions inside {}
        self.literal_parts = literal_parts  # Literal string parts
    
    def __repr__(self):
        return f'(FString {self.literal_parts} with {self.parts})'
    
    def variables(self):
        result = set()
        for part in self.parts:
            result.update(part.variables())
        return result


class Decorator(Expr):
    """Decorator expression."""
    def __init__(self, name: str, args: List[Expr] = None):
        self.name = name
        self.args = args or []  # Decorator arguments
    
    def __repr__(self):
        if self.args:
            args_str = ', '.join(str(a) for a in self.args)
            return f'(@{self.name}({args_str}))'
        return f'(@{self.name})'
    
    def variables(self):
        result = set()
        for arg in self.args:
            result.update(arg.variables())
        return result


class DecoratorChain(Expr):
    """Decorator chain (multiple decorators)."""
    def __init__(self, decorators: List[Decorator]):
        self.decorators = decorators
    
    def __repr__(self):
        dec_str = '\n'.join(str(d) for d in self.decorators)
        return f'(DecoratorChain\n{dec_str})'
    
    def variables(self):
        result = set()
        for dec in self.decorators:
            result.update(dec.variables())
        return result


class DataClass(Expr):
    """Data class decorator (@dataclass)."""
    def __init__(self, name: str, fields: List[tuple] = None, 
                 init: bool = True, repr: bool = True, eq: bool = True):
        self.name = name
        self.fields = fields or []  # (field_name, type, default)
        self.init = init
        self.repr = repr
        self.eq = eq
    
    def __repr__(self):
        fields_str = ', '.join(f'{name}: {ty}' for name, ty, _ in self.fields)
        return f'(DataClass {self.name} fields=[{fields_str}])'
    
    def variables(self):
        return set()


class TypeAlias(Expr):
    """Type alias definition."""
    def __init__(self, name: str, target: 'Type'):
        self.name = name
        self.target = target
    
    def __repr__(self):
        return f'(TypeAlias {self.name} = {self.target})'
    
    def variables(self):
        return set()


class Protocol(Expr):
    """Protocol definition (structural subtyping)."""
    def __init__(self, name: str, methods: List['MethodSignature'] = None,
                 attributes: List[tuple] = None):
        self.name = name
        self.methods = methods or []  # Method signatures
        self.attributes = attributes or []  # Attribute signatures
    
    def __repr__(self):
        return f'(Protocol {self.name})'
    
    def variables(self):
        return set()


class MethodSignature(Expr):
    """Method signature in protocol."""
    def __init__(self, name: str, params: List[tuple], returns: 'Type' = None):
        self.name = name
        self.params = params
        self.returns = returns
    
    def __repr__(self):
        return f'(MethodSignature {self.name})'
    
    def variables(self):
        return set()


class TypeVar(Expr):
    """Type variable for generics."""
    def __init__(self, name: str, bound: 'Type' = None, covariant: bool = False,
                 contravariant: bool = False):
        self.name = name
        self.bound = bound
        self.covariant = covariant
        self.contravariant = contravariant
    
    def __repr__(self):
        return f'(TypeVar {self.name})'
    
    def variables(self):
        return set()


class UnionType(Expr):
    """Union type (X | Y | Z)."""
    def __init__(self, types: List['Type']):
        self.types = types
    
    def __repr__(self):
        return f'({" | ".join(str(t) for t in self.types)})'
    
    def variables(self):
        return set()


class OptionalType(Expr):
    """Optional type (X | None)."""
    def __init__(self, type_expr: 'Type'):
        self.type_expr = type_expr
    
    def __repr__(self):
        return f'(Optional {self.type_expr})'
    
    def variables(self):
        return set()


class LiteralType(Expr):
    """Literal type (e.g., Literal[1, "a", True])."""
    def __init__(self, values: List[Expr]):
        self.values = values
    
    def __repr__(self):
        return f'(Literal [{" ".join(str(v) for v in self.values)}])'
    
    def variables(self):
        result = set()
        for v in self.values:
            result.update(v.variables())
        return result


class Final(Expr):
    """Final constant qualifier."""
    def __init__(self, name: str, value: Expr):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f'(Final {self.name} = {self.value})'
    
    def variables(self):
        return self.value.variables()


class TypeGuard(Expr):
    """Type guard assertion."""
    def __init__(self, expr: Expr, target_type: 'Type'):
        self.expr = expr
        self.target_type = target_type
    
    def __repr__(self):
        return f'(TypeGuard {self.expr} is {self.target_type})'
    
    def variables(self):
        return self.expr.variables()
