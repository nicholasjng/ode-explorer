import numpy as np
from ode_explorer.integrator import Integrator
from ode_explorer.model import ODEModel

from typing import Dict, Text, Any, Callable


class Metric:
    def __call__(self,
                 i: int,
                 integrator: Integrator,
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> Any:
        raise NotImplementedError


class DistanceToSolution(Metric):
    """
    Tracks distance of an ODE state vector to the state vector of a known
    solution. This is useful to test whether an integrator performs as
    expected.
    """
    def __init__(self, solution: Callable, norm: int = None,
                 name: Text = None):

        self.solution = solution
        self.norm = norm or 2
        self.name = name or "solution_distance"

    def __call__(self,
                 i: int,
                 integrator: Integrator,
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> Any:

        updated_state = locals["updated_state_dict"]

        t = updated_state[model.indep_name]
        y = np.array([updated_state[key] for key in model.dim_names])

        y_pred = self.solution(t)

        return np.linalg.norm(y - y_pred)