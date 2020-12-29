from typing import Dict, Text, Any, Callable, Union

import numpy as np

from ode_explorer.models import ODEModel
from ode_explorer.types import ModelState


class Metric:

    def __init__(self, name: Text = None):
        self.__name__ = name or self.__class__.__name__

    def __call__(self,
                 i: int,
                 state: ModelState,
                 updated_state: ModelState,
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> Any:
        raise NotImplementedError


class DistanceToSolution(Metric):
    """
    Tracks distance of an ODE state vector to the state vector of a known
    solution. This is useful to test whether an integrator performs as
    expected.
    """

    def __init__(self,
                 solution: Callable,
                 norm: Union[Text, int] = None,
                 name: Text = None):
        super(DistanceToSolution, self).__init__(name=name)

        self.solution = solution
        self.norm = norm or None

    def __call__(self,
                 i: int,
                 state: ModelState,
                 updated_state: ModelState,
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> Any:
        t, y = updated_state

        y_pred = self.solution(t)

        return np.linalg.norm(y - y_pred, ord=self.norm)
