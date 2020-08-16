
import numpy as np
from typing import Callable, Dict, Any, Text
from ode_explorer.model import ODEModel


class StepFunction:
    """
    Base class for all ODE step functions.
    """
    def __init__(self):

        # order of the method
        self.order = 0

    def forward(self,
                ode_fn: ODEModel,
                state: Dict[Text, float]) -> Dict[Text, float]:
        raise NotImplementedError
