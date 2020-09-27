import numpy as np
from typing import Callable, Dict, Any, Text, List, Union
from utils.import_utils import import_func_from_module


class ODEModel:
    """
    Base class for all ODE models.
    """
    # TODO: Make a function to infer the model dimension from the initial state
    def __init__(self,
                 module_path: Text = None,
                 ode_fn_name: Text = None,
                 ode_fn: Callable[[float, Union[float, np.ndarray], Any],
                                  Union[float, np.ndarray]] = None,
                 fn_args: Dict[Text, Any] = None,
                 indep_name: Text = None,
                 variable_name: Text = None,
                 dim_names: List[Text] = None):

        # ODE function, right hand side of y' = f(t,y)
        if not any([bool(module_path), bool(ode_fn_name),
                    bool(ode_fn)]):
            raise ValueError("Missing model information. Supply a right hand "
                             "side f(t,y) either by specifying a source path "
                             "or a callable function.")

        if any([bool(module_path), bool(ode_fn_name)]) and bool(ode_fn):
            raise ValueError("Defining a model function by a source path and "
                             "by a callable function object are mutually "
                             "exclusive. Please supply only one of these "
                             "options.")

        if bool(ode_fn):
            self.ode_fn = ode_fn
        else:
            self.ode_fn = import_func_from_module(module_path, ode_fn_name)

        # additional arguments for the function
        self.fn_args = fn_args

        self.indep_name = indep_name or "t"
        self.variable_name = variable_name or "y"
        self.dim_names = dim_names

    def update_args(self, **kwargs):
        if self.fn_args:
            self.fn_args.update(kwargs)
        else:
            self.fn_args = kwargs

    def make_initial_state(self, initial_time: float, initial_vec: Any):
        return {self.indep_name: initial_time, self.variable_name: initial_vec}

    def initialize_dim_names(self, initial_state: Dict[Text, Any]):
        if not any(hasattr(v, "__len__") for v in initial_state.values()):
            num_dims = len(list(initial_state.keys())) - 1
        else:
            num_dims = -1  # account for time
            for v in initial_state.values():
                if isinstance(v, float):
                    num_dims += 1
                else:
                    # this implicitly allows only list-likes
                    num_dims += len(v)

        # TODO: Graceful error handling by renaming?
        if self.dim_names and len(self.dim_names) != num_dims:
            raise ValueError("Error: Dimension mismatch. List of dimension "
                             "names suggests a system of size {0}, but "
                             "inferred system size {1} from initial state."
                             "".format(len(self.dim_names), num_dims))

        if not self.dim_names:
            if num_dims == 1:
                self.dim_names = list(self.variable_name)
            else:
                self.dim_names = ["{0}_{1}".format(self.variable_name, i)
                                  for i in range(1, num_dims + 1)]

    def __call__(self, t: float, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param t: Float, time index.
        :param y: np.ndarray, state of the ODE system at time t.
        :param kwargs: Additional keyword arguments to the ODE function.
        :return: Dict with keys as variable names and float values as state
        values at time t.
        """
        if kwargs:
            self.update_args(**kwargs)

        return self.ode_fn(t, y, **self.fn_args)
