from z3 import Int, Solver, sat

class Veripy:
    def __init__(self):
        self.config = {
            "enable_verification": True,
            "enable_runtime_checks": True
        }

    def require(self, condition):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self.config["enable_verification"]:
                    solver = Solver()
                    variables = {f'x{i}': Int(f'x{i}') for i in range(len(args))}
                    for var_name, value in zip(variables.keys(), args):
                        solver.add(variables[var_name] == value)

                    solver.add(condition(variables))

                    if solver.check() != sat:
                        raise AssertionError("Precondition failed")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def ensure(self, condition):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if self.config["enable_verification"]:
                    solver = Solver()
                    variables = {'result': Int('result')}
                    solver.add(variables['result'] == result)

                    solver.add(condition(variables))

                    if solver.check() != sat:
                        raise AssertionError("Postcondition failed")

                return result
            return wrapper
        return decorator

veripy = Veripy()

@veripy.require(lambda vars: vars['x0'] > 0)
@veripy.ensure(lambda vars: vars['result'] > 0)
def example_func_with_api(x):
    return x * 2


veripy.config["enable_verification"] = True
assert example_func_with_api(5) == 10  

veripy.config["enable_verification"] = False
assert example_func_with_api(-5) == -10  