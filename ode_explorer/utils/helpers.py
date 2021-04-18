import inspect
from typing import Callable, List, Text

__all__ = ["infer_variable_names", "infer_separability"]


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
    if {"t", "y"}.issubset(arg_set):
        return ["t", "y"]
    elif {"t", "q", "p"}.issubset(arg_set):
        return ["t", "q", "p"]
    else:
        # try to infer the variable names as the positional arguments
        num_pos = len(ode_spec.defaults)

        if num_args >= num_pos + 2:
            return args[:-num_pos]
        else:
            raise ValueError("Incompatible function signature for ODE integration.")


def infer_separability(q_derivative: Callable, p_derivative: Callable) -> bool:
    """
    Infer whether a Hamiltonian is separable.

    Args:
        q_derivative: Function returning the (vector-valued) q-derivative of the Hamiltonian.
        p_derivative: Function returning the (vector-valued) p-derivative of the Hamiltonian.

    Returns:
        A boolean indicating whether or not the Hamiltonian is separable based on its derivatives.
    """

    is_separable = False

    q_set = set(infer_variable_names(q_derivative))
    p_set = set(infer_variable_names(p_derivative))

    # TODO: Devise more / better checks than just derivative signatures
    if "q" not in p_set and "p" not in q_set:
        is_separable = True

    return is_separable
