from typing import Dict, Any, Text, List, Callable

from ode_explorer.constants import ModelMetadataKeys
from ode_explorer.models import BaseModel
from ode_explorer.models import messages
from ode_explorer.types import StateVariable
from ode_explorer.utils.helpers import infer_variable_names
from ode_explorer.utils.import_utils import import_func_from_module

ODEFunction = Callable[[StateVariable, StateVariable, Any], StateVariable]


class ODEModel(BaseModel):
    """
    Base class for all ODE models.

    An ODEModel implements the right-hand side (RHS) ``f`` of an ordinary differential equation ::

        y'(t) = f(t, y).

    In addition to the actual right-hand side, the ODEModel class keeps a minimal amount of state around,
    mainly for bookkeeping and easier visualization using pandas.

    Attributes:
        ode_fn: Right-hand side of the ODE.
        fn_args: Dict with additional keyword arguments for the ode_fn.
        variable_names: List of ODE variable names, taken from the signature of the ode_fn.
        dim_names: Optional list of dimension names for result data saving. These will become column
         headers in result pandas.DataFrame objects.
    """

    def __init__(self,
                 ode_fn: ODEFunction = None,
                 module_path: Text = None,
                 ode_fn_name: Text = None,
                 fn_args: Dict[Text, Any] = None,
                 dim_names: List[Text] = None) -> None:
        """
        ODEModel constructor.

        Args:
            ode_fn: Callable implementing the right-hand side of the model.
            module_path: Optional, path to a module where the Hamiltonian and its derivatives are
             defined. May be used instead of the direct function definition.
            ode_fn_name: Name of the function to be used as ode_fn. Needs to be present in the module file
             specified in the module_path argument.
            fn_args: Additional keyword arguments for ode_fn.
            dim_names: Optional list of dimension names for result data saving. These will become column
             headers in result pandas.DataFrame objects.
        """
        if not any([bool(module_path), bool(ode_fn_name), bool(ode_fn)]):
            raise ValueError(messages.MISSING_INFO)

        if any([bool(module_path), bool(ode_fn_name)]) and bool(ode_fn):
            raise ValueError(messages.BAD_MODEL_DEF)

        if bool(ode_fn):
            self.ode_fn = ode_fn
        else:
            self.ode_fn = import_func_from_module(module_path, ode_fn_name)

        # additional arguments for the function
        self.fn_args = fn_args or {}

        self.variable_names = infer_variable_names(rhs=ode_fn)
        self.dim_names = dim_names or []

    def update_args(self, **kwargs):
        """
        Update the model's keyword arguments.

        Args:
            **kwargs: Updated keyword arguments to replace the old ones.
        """
        self.fn_args.update(kwargs)

    def make_state(self, t: StateVariable, y: StateVariable):
        """
        Constructs a state object from raw input floats and numpy arrays.

        Args:
            t: Time variable at the current state.
            y: Spatial variable at the current state.

        Returns:
            A state object representing the current model state.
        """
        return t, y

    def get_metadata(self):
        """
        Return model metadata information. Used for constructing result pandas DataFrame objects.

        Returns:
            A dict with model metadata information.
        """

        return {ModelMetadataKeys.VARIABLE_NAMES: self.variable_names,
                ModelMetadataKeys.DIM_NAMES: self.dim_names}

    def __call__(self, t: StateVariable, y: StateVariable) -> StateVariable:
        """
        ODE model call operator.

        Args:
            t: Time variable at the current state.
            y: Spatial variable at the current state.

        Returns:
            A spatial variable representing the right-hand side given by the ode_fn
             at the input state.

        """
        return self.ode_fn(t, y, **self.fn_args)
