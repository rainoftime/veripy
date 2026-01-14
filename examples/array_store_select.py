from z3 import (
    Bool,
    BoolVal,
    Const,
    Int,
    IntVal,
    Solver,
    sat,
    is_expr,
    simplify,
    Select,
    Store,
)

def require(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            solver = Solver()
            variables = {}
            for i, value in enumerate(args):
                var_name = f"x{i}"
                if is_expr(value):
                    var = Const(var_name, value.sort())
                    coerced_value = value
                elif isinstance(value, bool):
                    var = Bool(var_name)
                    coerced_value = BoolVal(value)
                elif isinstance(value, int):
                    var = Int(var_name)
                    coerced_value = IntVal(value)
                else:
                    raise TypeError(
                        f"Unsupported argument type for {var_name}: {type(value)!r}. "
                        "Pass a Z3 expression/array or a Python int/bool."
                    )

                variables[var_name] = var
                solver.add(var == coerced_value)

            solver.add(condition(variables))

            if solver.check() == sat:
                return func(*args, **kwargs)
            else:
                raise AssertionError("Precondition failed")

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
                raise AssertionError("Postcondition failed")

        return wrapper
    return decorator

@require(lambda vars: vars['x0'] > 0)
@ensure(lambda vars: vars['result'] > 0)
def example_func_with_array(x, arr):
    array = Store(arr, 0, x)  
    return Select(array, 0)  


from z3 import Array, IntSort


array_var = Array('array', IntSort(), IntSort())
assert simplify(example_func_with_array(5, array_var)).as_long() == 5
try:
    example_func_with_array(-5, array_var)  
except AssertionError:
    print("Precondition or postcondition test passed")