from typing import Dict, Any, Text, List, Callable
from ode_explorer.models import BaseModel
from ode_explorer.types import ModelState, StateVariable
from ode_explorer.models import messages
from ode_explorer.utils.helpers import is_scalar, infer_variable_names, infer_separability
from ode_explorer.utils.import_utils import import_func_from_module

Hamiltonian = Callable[[ModelState], float]


class HamiltonianSystem(BaseModel):
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
            self.is_separable = infer_separability(self.q_derivative,
                                                   self.p_derivative)

    def make_state(self, t: StateVariable, q: StateVariable, p: StateVariable):
        return t, q, p

    def update_args(self, **kwargs):
        self.h_args.update(kwargs)

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

    def __call__(self, t: StateVariable, q: StateVariable, p: StateVariable) -> float:
        return self.hamiltonian(t, q, p, **self.h_args)
