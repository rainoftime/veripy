"""Value types for veripy syntax."""


class Value:
    """Base class for literal values."""
    __slots__ = ['v']
    
    def __init__(self, v):
        self.v = v
    
    def __eq__(self, other):
        return type(self) is type(other) and self.v == other.v
    
    def __hash__(self):
        return hash((type(self), self.v))


class VInt(Value):
    """Integer value."""
    def __init__(self, v):
        super().__init__(int(v))
    
    def __str__(self):
        return f'VInt {self.v}'
    
    def __repr__(self):
        return f'VInt {self.v}'


class VBool(Value):
    """Boolean value."""
    def __init__(self, v):
        super().__init__(v == 'True' or v == True)
    
    def __str__(self):
        return f'VBool {self.v}'
    
    def __repr__(self):
        return f'VBool {self.v}'


class VString(Value):
    """String literal value."""
    def __init__(self, v):
        super().__init__(str(v))
    
    def __str__(self):
        return f'VString {self.v}'
    
    def __repr__(self):
        return f'VString {self.v}'


class VSet(Value):
    """Set literal value."""
    def __init__(self, elements=None):
        super().__init__(set(elements) if elements else set())
    
    def __str__(self):
        return f'VSet {self.v}'
    
    def __repr__(self):
        return f'VSet {self.v}'


class VDict(Value):
    """Dictionary literal value."""
    def __init__(self, pairs=None):
        super().__init__(dict(pairs) if pairs else {})
    
    def __str__(self):
        return f'VDict {self.v}'
    
    def __repr__(self):
        return f'VDict {self.v}'


class VList(Value):
    """List literal value."""
    def __init__(self, elements=None):
        super().__init__(list(elements) if elements else [])
    
    def __str__(self):
        return f'VList {self.v}'
    
    def __repr__(self):
        return f'VList {self.v}'
