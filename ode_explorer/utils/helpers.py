import inspect
from typing import Callable, List, Text

from ode_explorer.defaults import standard_rhs, hamiltonian_rhs

__all__ = ["is_scalar", "infer_variable_names", "infer_separability"]


def is_scalar(y):
    """
    Infer whether an ODE is scalar.

    Args:
        y: State vector.

    Returns:
        A boolean, True if the ODE state vector is scalar and False otherwise.
    """

    return not hasattr(y, "__len__")


def infer_variable_names(rhs: Callable) -> List[Text]:
    """
    Infer the variable names from the right-hand side function of an ODE model.

    Args:
        rhs: Right-hand side to infer variable names from.

    Returns:
        A list containing the ODE variable names.
    """

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


def infer_separability(q_derivative: Callable, p_derivative: Callable) -> bool:
    """
    Infer whether a Hamiltonian is separable.

    Args:
        q_derivative: Function returning the (vector-valued) q-derivative of the Hamiltonian.
        p_derivative: Function returning the (vector-valued) p-derivative of the Hamiltonian.

    Returns:
        A boolean indicating whether the Hamiltonian is separable based on its derivatives or not.
    """

    is_separable = False

    q_set = set(infer_variable_names(q_derivative))
    p_set = set(infer_variable_names(p_derivative))

    # TODO: Devise more / better checks than just derivative signatures
    if "q" not in p_set and "p" not in q_set:
        is_separable = True

    return is_separable
