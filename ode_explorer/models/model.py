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
    """

    def __init__(self,
                 ode_fn: ODEFunction = None,
                 module_path: Text = None,
                 ode_fn_name: Text = None,
                 fn_args: Dict[Text, Any] = None,
                 dim_names: List[Text] = None):

        # ODE function, right hand side of y' = f(t,y)
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
        self.fn_args.update(kwargs)

    def make_state(self, time: StateVariable, vec: StateVariable):
        return time, vec

    def get_metadata(self):

        return {ModelMetadataKeys.VARIABLE_NAMES: self.variable_names,
                ModelMetadataKeys.DIM_NAMES: self.dim_names}

    def __call__(self, t: StateVariable, y: StateVariable) -> StateVariable:

        return self.ode_fn(t, y, **self.fn_args)
