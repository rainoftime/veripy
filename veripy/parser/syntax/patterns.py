"""Pattern matching classes for veripy syntax."""

from typing import List
from .statements import Stmt
from .expressions import Expr


class Match(Stmt):
    """Match statement (Python 3.10+)."""
    def __init__(self, subject: Expr, cases: List['MatchCase']):
        self.subject = subject
        self.cases = cases  # List of MatchCase
    
    def __repr__(self):
        cases_str = ', '.join(str(c) for c in self.cases)
        return f'(Match {self.subject} cases=[{cases_str}])'
    
    def variables(self):
        result = self.subject.variables()
        for case in self.cases:
            result.update(case.body.variables())
        return result


class MatchCase:
    """Single case in match statement."""
    def __init__(self, pattern: 'Pattern', guard: Expr = None, body: Stmt = None):
        self.pattern = pattern
        self.guard = guard  # Optional guard expression
        self.body = body  # Case body
    
    def __repr__(self):
        guard_str = f' if {self.guard}' if self.guard else ''
        return f'(Case {self.pattern}{guard_str} => {self.body})'


class Pattern:
    """Base class for match patterns."""
    pass


class PatternConstant(Pattern):
    """Constant pattern (literals)."""
    def __init__(self, value: Expr):
        self.value = value
    
    def __repr__(self):
        return f'(PatternConstant {self.value})'


class PatternCapture(Pattern):
    """Capture pattern (variable binding)."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f'(PatternCapture {self.name})'


class PatternSequence(Pattern):
    """Sequence pattern (tuple/list)."""
    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
    
    def __repr__(self):
        return f'(PatternSequence [{", ".join(str(p) for p in self.patterns)}])'


class PatternMapping(Pattern):
    """Mapping pattern (dict)."""
    def __init__(self, keys: List[Expr], patterns: List[Pattern], rest: str = None):
        self.keys = keys
        self.patterns = patterns
        self.rest = rest  # Optional **rest binding
    
    def __repr__(self):
        pairs = [f'{k}: {p}' for k, p in zip(self.keys, self.patterns)]
        if self.rest:
            pairs.append(f'**{self.rest}')
        return f'(PatternMapping {{{", ".join(pairs)}}})'


class PatternClass(Pattern):
    """Class pattern (object matching)."""
    def __init__(self, class_name: str, patterns: List[Pattern]):
        self.class_name = class_name
        self.patterns = patterns
    
    def __repr__(self):
        return f'(PatternClass {self.class_name}({", ".join(str(p) for p in self.patterns)}))'


class PatternLiteral(Pattern):
    """Literal pattern (constants)."""
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f'(PatternLiteral {self.value})'


class PatternAs(Pattern):
    """As pattern (alias)."""
    def __init__(self, pattern: Pattern, name: str):
        self.pattern = pattern
        self.name = name
    
    def __repr__(self):
        return f'(PatternAs {self.pattern} as {self.name})'


class PatternOr(Pattern):
    """Or pattern (alternative)."""
    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
    
    def __repr__(self):
        return f'(PatternOr {" | ".join(str(p) for p in self.patterns)})'
