
import numpy as np
from typing import Callable, Dict, Any, Text, List


class ODEModel:
    """
    Base class for all ODE models.
    """
    def __init__(self,
                 ode_fn: Callable[[float, np.ndarray, Dict[Text, Any]],
                                  np.ndarray],
                 fn_args: Dict[Text, Any] = None,
                 variable_names: List[Text] = None,
                 indep_name: Text = None,
                 ):
        # ODE function, right hand side of y' = f(t,y)
        self.ode_fn = ode_fn

        # additional arguments for the function
        self.fn_args = fn_args

        # state variable names for DataFrame columns
        if not variable_names:
            self.variable_names = []
        else:
            self.variable_names = variable_names

        self.model_dim = len(variable_names)

        if not indep_name:
            self.indep_name = "time"
        else:
            self.indep_name = indep_name

    def update_args(self, **kwargs):

        if self.fn_args:
            self.fn_args.update(kwargs)
        else:
            self.fn_args = kwargs

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

        ynew = self.ode_fn(t, y, **self.fn_args)

        if not len(ynew) == len(self.variable_names):
            raise ValueError("Error: Variable names and ODE system size "
                             "do not match.")

        return ynew
