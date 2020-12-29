import inspect
from typing import Callable

from ode_explorer.defaults import standard_rhs, hamiltonian_rhs

__all__ = ["is_scalar", "infer_variable_names", "infer_separability"]


def is_scalar(y):
    return not hasattr(y, "__len__")


def infer_variable_names(rhs: Callable):
    ode_spec = inspect.getfullargspec(func=rhs)

    args = ode_spec.args

    num_args, arg_set = len(args), set(args)

    # check if the function spec is either of the standard ones
    # if true, return them
    if set(standard_rhs).issubset(arg_set):
        return standard_rhs
    elif set(hamiltonian_rhs).issubset(arg_set):
        return hamiltonian_rhs
    else:
        # try to infer the variable names as those without defaults
        num_defaults = len(ode_spec.defaults)

        if num_args >= num_defaults + 2:
            return args[:-num_defaults]


def infer_separability(q_derivative: Callable,
                       p_derivative: Callable) -> bool:
    is_separable = False

    # h_set = set(infer_variable_names(hamiltonian))
    q_set = set(infer_variable_names(q_derivative))
    p_set = set(infer_variable_names(p_derivative))

    # TODO: Devise more / better checks than just derivative
    #  signatures
    if "q" not in p_set and "p" not in q_set:
        is_separable = True

    return is_separable
