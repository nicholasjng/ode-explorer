import numpy as np
from typing import Callable, Dict, Any, Text, List
from utils.import_utils import import_func_from_module


class ODEModel:
    """
    Base class for all ODE models.
    """
    # TODO: Make a function to infer the model dimension from the initial state
    def __init__(self,
                 module_path: Text = None,
                 ode_fn_name: Text = None,
                 ode_fn: Callable[[float, np.ndarray, Dict[Text, Any]],
                                  np.ndarray] = None,
                 fn_args: Dict[Text, Any] = None,
                 ode_dimension: int = None,
                 variable_names: List[Text] = None,
                 indep_name: Text = None,
                 ):
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

        if not any([bool(variable_names), bool(ode_dimension)]):
            raise ValueError("Please specify information about the "
                             "dimensionality of your ODE system, either "
                             "by the \"ode_dimension\" or \"variable_names\" "
                             "arguments.")

        if bool(ode_fn):
            self.ode_fn = ode_fn
        else:
            self.ode_fn = import_func_from_module(module_path, ode_fn_name)

        # additional arguments for the function
        self.fn_args = fn_args

        # state variable names for DataFrame columns
        if not variable_names:
            self.variable_names = []
        else:
            self.variable_names = variable_names

        self.model_dim = len(self.variable_names)

        if not indep_name:
            self.indep_name = "time"
        else:
            self.indep_name = indep_name

    def update_args(self, **kwargs):
        if self.fn_args:
            self.fn_args.update(kwargs)
        else:
            self.fn_args = kwargs

    def register_equation_data(self, initial_state: Dict[Text, Any]):
        try:
            y = initial_state["y"]
        except Exception as e:
            print(e)
            return

        model_dim = len(y)
        if not self.variable_names:
            self.variable_names = ["y_{}".format(i) for i in range(model_dim)]

    def __call__(self, t: float, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param t: Float, time index.
        :param y: np.ndarray, state of the ODE system at time t.
        :param kwargs: Additional keyword arguments to the ODE function.
        :return: Dict with keys as variable names and float values as state
        values at time t.
        """
        if not len(y) == len(self.variable_names):
            raise ValueError("Error: Variable names and ODE system size "
                             "do not match.")

        if kwargs:
            self.update_args(**kwargs)

        ynew = self.ode_fn(t, y, **self.fn_args)

        return ynew
