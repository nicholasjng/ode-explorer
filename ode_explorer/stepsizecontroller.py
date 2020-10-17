from ode_explorer.integrator import Integrator
from ode_explorer.model import ODEModel
from typing import Dict, Text, Any


class StepsizeController:
    def __call__(self,
                 i: int,
                 integrator: Integrator,
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> float:
        raise NotImplementedError


