from typing import Dict, Any, Text, List

from ode_explorer.constants import ModelMetadataKeys
from ode_explorer.models import messages
from ode_explorer.models.base_model import BaseModel
from ode_explorer.types import ModelState, StateVariable
from ode_explorer.types import ODEFunction
from ode_explorer.utils.helpers import is_scalar, infer_variable_names
from ode_explorer.utils.import_utils import import_func_from_module


class ODEModel(BaseModel):
    """
    Base class for all ODE models.
    """

    def __init__(self,
                 module_path: Text = None,
                 ode_fn_name: Text = None,
                 ode_fn: ODEFunction = None,
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

        self.variable_names = infer_variable_names(ode_fn=ode_fn)
        self.dim_names = dim_names or []

    def update_args(self, **kwargs):
        self.fn_args.update(kwargs)

    def make_state(self, time: StateVariable, vec: StateVariable):
        return time, vec

    def initialize_dim_names(self, initial_state: ModelState):

        var_dims = []

        for k, v in zip(self.variable_names, initial_state):
            dim = 1 if is_scalar(v) else len(v)

            var_dims.append((k, dim))

        num_dims = sum(v[-1] for v in var_dims)

        # TODO: Graceful error handling by renaming?
        if self.dim_names and len(self.dim_names) != num_dims:
            raise ValueError(messages.DIMENSION_MISMATCH.format(len(self.dim_names), num_dims))

        if not self.dim_names:
            dim_names = []
            for i, (name, dim) in enumerate(var_dims):
                if dim == 1:
                    dim_names += [name]
                else:
                    dim_names += ["{0}_{1}".format(name, i) for i in range(1, dim + 1)]

            self.dim_names = dim_names

    def get_metadata(self):

        return {ModelMetadataKeys.VARIABLE_NAMES: self.variable_names,
                ModelMetadataKeys.DIM_NAMES: self.dim_names}

    def __call__(self, t: StateVariable, y: StateVariable) -> StateVariable:

        return self.ode_fn(t, y, **self.fn_args)
