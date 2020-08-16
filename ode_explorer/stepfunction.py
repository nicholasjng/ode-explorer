import numpy as np
from typing import Dict, Text
from ode_explorer.model import ODEModel


class StepFunction:
    """
    Base class for all ODE step functions.
    """
    def __init__(self):

        # order of the method
        self.order = 0

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:
        raise NotImplementedError


class EulerMethod(StepFunction):
    """
    Euler method for ODE integration.
    """
    def __init__(self):

        super().__init__()
        self.order = 1

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:

        # copy here, might be expensive
        data = state.copy()
        t = state.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        y = np.array(data.values())

        y_new = y + h * model(t, y, **kwargs)

        new_state = {**{model.indep_name: t + h},
                     **dict(zip(model.variable_names, y_new))}

        return new_state
