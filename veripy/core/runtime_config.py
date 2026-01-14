from z3 import Int, Solver, sat

class VeripyConfig:
    ENABLE_VERIFICATION = True
    ENABLE_RUNTIME_CHECKS = True

class RuntimeConfig:
    """Runtime configuration for veripy verification."""
    def __init__(self):
        self.enable_verification = True
        self.enable_runtime_checks = True

# Global config instance
_config = RuntimeConfig()

def get_config():
    """Get the global runtime configuration."""
    return _config

def set_config(**kwargs):
    """Set runtime configuration options."""
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise AttributeError(f"Unknown config option: {key}")

def require(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if VeripyConfig.ENABLE_VERIFICATION:
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

def ensure(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if VeripyConfig.ENABLE_VERIFICATION:
                solver = Solver()
                variables = {'result': Int('result')}
                solver.add(variables['result'] == result)

                solver.add(condition(variables))

                if solver.check() != sat:
                    raise AssertionError("Postcondition failed")

            return result
        return wrapper
    return decorator

@require(lambda vars: vars['x0'] > 0)
@ensure(lambda vars: vars['result'] > 0)
def example_func_with_config(x):
    return x * 2


VeripyConfig.ENABLE_VERIFICATION = True
assert example_func_with_config(5) == 10  

VeripyConfig.ENABLE_VERIFICATION = False
assert example_func_with_config(-5) == -10  