import numpy as np
from ode_explorer.model import ODEModel
from typing import Dict, Text, Any, Union, Tuple


class StepsizeController:
    def __call__(self,
                 i: int,
                 h: float,
                 state: Dict[Text, Union[np.ndarray, float]],
                 updated_state: Dict[Text, Union[np.ndarray, float]],
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> Tuple[bool, float]:
        raise NotImplementedError


class DOPRI45Controller(StepsizeController):
    def __init__(self, fac_min: float, fac_max: float, tol: float,
                 safety_factor: float = 0.9):
        self.fac_min = fac_min
        self.fac_max = fac_max
        self.tol = tol
        self.safety_factor = safety_factor
        self.order = 5

    def __call__(self,
                 i: int,
                 h: float,
                 state: Dict[Text, Union[np.ndarray, float]],
                 updated_state: Dict[Text, Union[np.ndarray, float]],
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> Tuple[bool, float]:

        order4, order5 = updated_state

        var_name = model.variable_names[0]

        y_4, y_5 = order4[var_name], order5[var_name]

        norm_diff45 = np.linalg.norm(y_4 - y_5)
        accept = norm_diff45 < self.tol

        error_est = (self.tol / h * norm_diff45) ** (1 / self.order)

        h_new = h * min(self.fac_max,
                        max(self.fac_min, self.safety_factor * error_est))

        if not accept:
            h_new = 0.5 * h

        return accept, h_new
