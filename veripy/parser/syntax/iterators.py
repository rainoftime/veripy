"""Iterator, comprehension, and generator classes for veripy syntax."""

from typing import List, Optional
from .expressions import Expr, Literal, Var
from .values import VInt


class Iterator(Expr):
    """Iterator protocol (__iter__, __next__)."""
    def __init__(self, source: Expr):
        self.source = source  # Iterable source
    
    def __repr__(self):
        return f'(Iterator {self.source})'
    
    def variables(self):
        return self.source.variables()


class Range(Expr):
    """Range expression (for iteration)."""
    def __init__(self, start: Expr = None, stop: Expr = None, step: Expr = None):
        self.start = start or Literal(VInt(0))
        self.stop = stop
        self.step = step or Literal(VInt(1))
    
    def __repr__(self):
        if self.step.variables() == {Literal(VInt(1)).variables()}:
            return f'(Range {self.start}..{self.stop})'
        return f'(Range {self.start}..{self.stop}..{self.step})'
    
    def variables(self):
        result = self.start.variables()
        if self.stop:
            result.update(self.stop.variables())
        result.update(self.step.variables())
        return result


class Enumerate(Expr):
    """Enumerate expression (index, value pairs)."""
    def __init__(self, iterable: Expr, start: Expr = None):
        self.iterable = iterable
        self.start = start or Literal(VInt(0))
    
    def __repr__(self):
        return f'(Enumerate {self.iterable})'
    
    def variables(self):
        result = self.iterable.variables()
        result.update(self.start.variables())
        return result


class Zip(Expr):
    """Zip expression (multiple iterables)."""
    def __init__(self, iterables: List[Expr]):
        self.iterables = iterables
    
    def __repr__(self):
        return f'(Zip {self.iterables})'
    
    def variables(self):
        result = set()
        for it in self.iterables:
            result.update(it.variables())
        return result


class Map(Expr):
    """Map expression (function application)."""
    def __init__(self, func: Expr, iterable: Expr):
        self.func = func
        self.iterable = iterable
    
    def __repr__(self):
        return f'(Map {self.func} {self.iterable})'
    
    def variables(self):
        result = self.func.variables()
        result.update(self.iterable.variables())
        return result


class Filter(Expr):
    """Filter expression (conditional selection)."""
    def __init__(self, predicate: Expr, iterable: Expr):
        self.predicate = predicate
        self.iterable = iterable
    
    def __repr__(self):
        return f'(Filter {self.predicate} {self.iterable})'
    
    def variables(self):
        result = self.predicate.variables()
        result.update(self.iterable.variables())
        return result


class Reduce(Expr):
    """Reduce expression (fold operation)."""
    def __init__(self, func: Expr, iterable: Expr, initial: Expr = None):
        self.func = func
        self.iterable = iterable
        self.initial = initial
    
    def __repr__(self):
        return f'(Reduce {self.func} {self.iterable})'
    
    def variables(self):
        result = self.func.variables()
        result.update(self.iterable.variables())
        if self.initial:
            result.update(self.initial.variables())
        return result


class Comprehension(Expr):
    """Generic comprehension expression."""
    def __init__(self, kind: str, element: Expr, generators: List['Generator']):
        self.kind = kind  # 'list', 'set', 'dict', 'generator'
        self.element = element
        self.generators = generators
    
    def __repr__(self):
        return f'({self.kind} comprehension: {self.element} for {self.generators})'
    
    def variables(self):
        result = self.element.variables()
        for gen in self.generators:
            result.update(gen.iterable.variables())
            result.discard(gen.target.name)
            if gen.predicate:
                result.update(gen.predicate.variables())
        return result


class Generator(Expr):
    """Generator clause in comprehension."""
    def __init__(self, target: Var, iterable: Expr, predicate: Expr = None):
        self.target = target  # Loop variable
        self.iterable = iterable
        self.predicate = predicate  # Optional if condition
    
    def __repr__(self):
        if self.predicate:
            return f'(Generator {self.target} in {self.iterable} if {self.predicate})'
        return f'(Generator {self.target} in {self.iterable})'
    
    def variables(self):
        result = self.iterable.variables()
        if self.predicate:
            result.update(self.predicate.variables())
        result.discard(self.target.name)
        return result


class Lambda(Expr):
    """Lambda expression."""
    def __init__(self, params: List[tuple], body: Expr):
        self.params = params  # List of (param_name, type)
        self.body = body
    
    def __repr__(self):
        params_str = ', '.join(p for p, _ in self.params)
        return f'(Lambda ({params_str}) => {self.body})'
    
    def variables(self):
        param_names = {name for name, _ in self.params}
        body_vars = self.body.variables()
        return body_vars - param_names


class Yield(Expr):
    """Yield expression for generators."""
    def __init__(self, value: Expr = None):
        self.value = value  # Value to yield (None for yield without value)
    
    def __repr__(self):
        if self.value:
            return f'(Yield {self.value})'
        return '(Yield)'
    
    def variables(self):
        return self.value.variables() if self.value else set()


class YieldFrom(Expr):
    """Yield from expression for generator delegation."""
    def __init__(self, iterable: Expr):
        self.iterable = iterable
    
    def __repr__(self):
        return f'(YieldFrom {self.iterable})'
    
    def variables(self):
        return self.iterable.variables()


class Walrus(Expr):
    """Named expression (walrus operator :=)."""
    def __init__(self, name: str, value: Expr):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f'({self.name} := {self.value})'
    
    def variables(self):
        result = self.value.variables()
        result.add(self.name)
        return result
