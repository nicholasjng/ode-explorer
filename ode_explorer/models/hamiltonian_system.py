from typing import Dict, Any, Text, List, Callable

from ode_explorer.constants import ModelMetadataKeys
from ode_explorer.models import BaseModel
from ode_explorer.models import messages
from ode_explorer.types import StateVariable
from ode_explorer.utils.helpers import infer_variable_names, infer_separability
from ode_explorer.utils.import_utils import import_func_from_module

Hamiltonian = Callable[[StateVariable, StateVariable, StateVariable, Any], float]


class HamiltonianSystem(BaseModel):
    """
    Hamiltonian system base class. Use this with special step functions
    for solving Hamiltonian problems.

    The Hamiltonian System is characterized by three functions:

        - The Hamiltonian itself, a function H(t, p, q, **h_args),
        - The q-derivative del H / del q (t, q, p, **h_args) of the Hamiltonian,
        - The p-derivative del H / del p (t, q, p, **h_args) of the Hamiltonian.

    These can be either specified directly as functions or as names, in which case they will be imported
    from a specified module file.

    IMPORTANT: Right now, separable Hamiltonian systems are the only supported type;
    these are functions of the type ::
            H(t, p, q) = T(p) + V(q).

    Separability is inferred automatically by checking the signatures of the q- and p-derivatives
    of the Hamiltonian; a separable Hamiltonian needs a q-derivative independent of p, and a p-derivative
    independent of q. To specify a separable Hamiltonian correctly, supply a q-derivative with signature
    (t, q, **h_args) and a p-derivative of signature (t, p, **h_args).
    """

    def __init__(self,
                 hamiltonian: Hamiltonian = None,
                 q_derivative: Callable = None,
                 p_derivative: Callable = None,
                 module_path: Text = None,
                 hamiltonian_name: Text = None,
                 q_derivative_name: Text = None,
                 p_derivative_name: Text = None,
                 h_args: Dict[Text, Any] = None,
                 dim_names: List[Text] = None,
                 is_separable: bool = None):
        """
        Hamiltonian system constructor.

        Args:
            hamiltonian: Callable, computes the Hamiltonian at a point in phase space.
            q_derivative: Callable, compute the q-(spatial) derivative at a point in phase space.
            p_derivative: Callable, compute the p-(momentum) derivative at a point in phase space.
            module_path: Optional, path to a module where the Hamiltonian and its derivatives are
             defined. May be used instead of the direct function definition.
            hamiltonian_name: Name of the Hamiltonian. Needs to be present in the module file
             specified in the module_path argument.
            q_derivative_name: Name of the Hamiltonian's spatial derivative. Needs to be present
             in the module file specified in the module_path argument.
            p_derivative_name: Name of the Hamiltonian's momentum derivative. Needs to be present
             in the module file specified in the module_path argument.
            h_args: Additional keyword arguments for calling the Hamiltonian.
            dim_names: Optional names for the spatial and momentum dimensions.
            is_separable: Boolean indicator, whether the Hamiltonian is separable or not. If not
             supplied, will be inferred on construction.
        """
        # ODE function, right hand side of (q', p') = dH(t, q, p)
        if not any([bool(hamiltonian), bool(q_derivative), bool(p_derivative)]):
            raise ValueError(messages.MISSING_INFO)

        if any([bool(module_path), bool(hamiltonian_name)]) and bool(hamiltonian):
            raise ValueError(messages.BAD_MODEL_DEF)

        if bool(hamiltonian):
            self.hamiltonian = hamiltonian
            self.q_derivative = q_derivative
            self.p_derivative = p_derivative
        else:
            self.hamiltonian = import_func_from_module(module_path, hamiltonian_name)
            self.q_derivative = import_func_from_module(module_path, q_derivative_name)
            self.p_derivative = import_func_from_module(module_path, p_derivative_name)

        # additional arguments for the function
        self.h_args = h_args or {}

        self.variable_names = infer_variable_names(rhs=hamiltonian)
        self.dim_names = dim_names or []

        if is_separable is not None:
            self.is_separable = is_separable
        else:
            self.is_separable = infer_separability(self.q_derivative, self.p_derivative)

    def make_state(self, t: StateVariable, q: StateVariable, p: StateVariable):
        """
        Constructs a state object from raw input floats and numpy arrays.

        Args:
            t: Time variable at the current state.
            q: Spatial variable at the current state.
            p: Momentum variable at the current state.

        Returns:
            A state object representing the current point in phase space.
        """
        return t, q, p

    def update_args(self, **kwargs):
        """
        Update the Hamiltonian's keyword arguments.

        Args:
            **kwargs: Updated keyword arguments to replace the old ones.
        """
        self.h_args.update(kwargs)

    def get_metadata(self):
        """
        Return model metadata information. Used for constructing result pandas DataFrame objects.

        Returns:
            A dict with model metadata information.
        """
        return {ModelMetadataKeys.VARIABLE_NAMES: self.variable_names,
                ModelMetadataKeys.DIM_NAMES: self.dim_names}

    def __call__(self, t: StateVariable, q: StateVariable, p: StateVariable) -> float:
        """
        Hamiltonian System call operator. Call a HamiltonianSystem object to return a value
        of the defining Hamiltonian at a certain point in phase space ("state").

        Args:
            t: Time variable at the current state.
            q: Spatial variable at the current state.
            p: Momentum variable at the current state.

        Returns:
            A scalar, the value of the Hamiltonian at the current state.
        """
        return self.hamiltonian(t, q, p, **self.h_args)
