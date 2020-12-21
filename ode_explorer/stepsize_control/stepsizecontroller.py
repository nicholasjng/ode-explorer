from typing import Dict, Text, Any, Tuple

import numpy as np

from ode_explorer.models.model import ODEModel
from ode_explorer.types import ModelState


class StepSizeController:
    def __call__(self,
                 i: int,
                 h: float,
                 state: ModelState,
                 updated_state: ModelState,
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> Tuple[bool, float]:
        raise NotImplementedError


class DOPRI45Controller(StepSizeController):
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
                 state: ModelState,
                 updated_state: ModelState,
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> Tuple[bool, float]:
        order4, order5 = updated_state

        y_prev = state[-1]

        y_4, y_5 = order4[-1], order5[-1]

        err_tol = self.atol + self.rtol * np.maximum(np.abs(y_prev), np.abs(y_5))

        err_ratio = np.linalg.norm(y_4 / err_tol)

        accept = err_ratio < 1.

        error_est = (1 / err_ratio) ** (1 / self.order)

        h_new = h * min(self.fac_max, max(self.fac_min, self.safety_factor * error_est))

        return accept, h_new
