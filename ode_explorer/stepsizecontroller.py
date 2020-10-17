import numpy as np
from ode_explorer.model import ODEModel
from typing import Dict, Text, Any, Union


class StepsizeController:
    def __call__(self,
                 i: int,
                 state_dict: Dict[Text, Union[np.ndarray, float]],
                 updated_state_dict: Dict[Text, Union[np.ndarray, float]],
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> float:
        raise NotImplementedError


