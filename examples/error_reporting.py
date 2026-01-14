from z3 import Int, Solver, Select, Store, sat

def require(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            solver = Solver()
            variables = {f'x{i}': Int(f'x{i}') for i in range(len(args))}
            for var_name, value in zip(variables.keys(), args):
                solver.add(variables[var_name] == value)

            solver.add(condition(variables))

            if solver.check() == sat:
                return func(*args, **kwargs)
            else:
                raise AssertionError(f"Precondition failed with model: {solver.model()}")

        return wrapper
    return decorator

def ensure(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            solver = Solver()
            variables = {'result': Int('result')}
            solver.add(variables['result'] == result)

            solver.add(condition(variables))

            if solver.check() == sat:
                return result
            else:
                raise AssertionError(f"Postcondition failed with model: {solver.model()}")

        return wrapper
    return decorator

@require(lambda vars: vars['x0'] > 0)
@ensure(lambda vars: vars['result'] > 0)
def example_func_with_error_reporting(x):
    return x - 10


try:
    example_func_with_error_reporting(-5)  
except AssertionError as e:
    print(e)