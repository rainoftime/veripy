import z3
from z3 import Int, Solver, Select, Store, Array, IntSort, sat, Const

def test_postconditions():
    def ensure(condition):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                solver = Solver()
                variables = {'result': Int('result')}
                solver.add(variables['result'] == result)

                solver.add(condition(variables))

                if solver.check() != sat:
                    raise AssertionError("Postcondition failed")

                return result
            return wrapper
        return decorator

    @ensure(lambda vars: vars['result'] > 0)
    def example_positive_output(x):
        return x * 2

    assert example_positive_output(5) == 10
    try:
        example_positive_output(-5)
    except AssertionError:
        print("Postcondition test passed for example_positive_output")

def test_complex_data_types():
    def require(condition):
        def decorator(func):
            def wrapper(*args, **kwargs):
                solver = Solver()
                variables = {}
                for i, value in enumerate(args):
                    name = f'x{i}'
                    if hasattr(value, "sort"):
                        variables[name] = Const(name, value.sort())
                    else:
                        variables[name] = Int(name)
                    solver.add(variables[name] == value)

                solver.add(condition(variables))

                if solver.check() != sat:
                    raise AssertionError("Precondition failed")

                result = func(*args, **kwargs)
                if isinstance(result, z3.ExprRef):
                    simplified = z3.simplify(result)
                    if z3.is_int_value(simplified):
                        return simplified.as_long()
                    return simplified
                return result
            return wrapper
        return decorator

    @require(lambda vars: vars['x0'] >= 0)
    def example_array_operations(x, arr):
        updated_array = Store(arr, 0, x)
        return Select(updated_array, 0)

    array_var = Array('array', IntSort(), IntSort())
    assert example_array_operations(5, array_var) == 5
    try:
        example_array_operations(-5, array_var)
    except AssertionError:
        print("Precondition test passed for example_array_operations")

if __name__ == "__main__":
    test_postconditions()
    test_complex_data_types()
    print("All tests passed.")