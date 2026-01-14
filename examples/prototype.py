

from z3 import Int, Solver, And


def require(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            solver = Solver()
            variable_map = {f'x{i}': val for i, val in enumerate(args)}
            for cond in condition:
               assert cond(variable_map)