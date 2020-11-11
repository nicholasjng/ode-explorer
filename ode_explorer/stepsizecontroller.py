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
    def __init__(self,
                 atol: float = 0.001,
                 rtol: float = 0.001,
                 fac_min: float = 0.2,
                 fac_max: float = 5.0,
                 safety_factor: float = 0.9):

        self.atol = atol
        self.rtol = rtol
        # maximal step size reduction factor
        self.fac_min = fac_min
        # maximal step size increase factor
        self.fac_max = fac_max
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

        # TODO: This only works for y' = f(t,y) type situations
        var_name = model.variable_names[0]

        y_prev = state[var_name]
        y_4, y_5 = order4[var_name], order5[var_name]

        err_tol = self.atol + self.rtol * np.maximum(np.abs(y_prev),
                                                     np.abs(y_5))

        err_ratio = np.linalg.norm(y_4 / err_tol)

        accept = err_ratio < 1.

        error_est = (1 / err_ratio) ** (1 / self.order)

        h_new = h * min(self.fac_max,
                        max(self.fac_min, self.safety_factor * error_est))

        return accept, h_new
