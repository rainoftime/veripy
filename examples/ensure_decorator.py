from z3 import Int, Solver, sat

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
                raise AssertionError("Postcondition failed")

        return wrapper
    return decorator

@ensure(lambda vars: vars['result'] > 0)
def example_func_output(x):
    return x * 2


assert example_func_output(5) == 10  
try:
    example_func_output(-5)  
except AssertionError:
    print("Postcondition test passed")