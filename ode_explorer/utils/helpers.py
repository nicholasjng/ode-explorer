import inspect
from typing import Callable

from ode_explorer import defaults

__all__ = ["is_scalar", "infer_variable_names"]


def is_scalar(y):
    return not hasattr(y, "__len__")


def infer_variable_names(ode_fn: Callable):
    standard_rhs = defaults.standard_rhs
    hamiltonian_rhs = defaults.hamiltonian_rhs

    ode_argspec = inspect.getfullargspec(func=ode_fn)

    args = ode_argspec.args

    num_args, arg_set = len(args), set(args)

    # check if the function spec is either of the standard ones
    # if true, return them
    if set(standard_rhs).issubset(arg_set):
        return standard_rhs
    elif set(hamiltonian_rhs).issubset(arg_set):
        return hamiltonian_rhs
    else:
        # try to infer the variable names as those without defaults
        num_defaults = len(ode_argspec.defaults)

        if num_args >= num_defaults + 2:
            return args[:-num_defaults]
