from typing import Dict, Text, Any, Tuple

import numpy as np

from ode_explorer.models.model import BaseModel
from ode_explorer.types import ModelState


class StepSizeController:
    """
    Base StepSizeController interface. Subclass this to define your own custom step size control
    functions.
    """

    def __call__(self,
                 i: int,
                 h: float,
                 state: ModelState,
                 updated_state: ModelState,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> Tuple[bool, float]:
        raise NotImplementedError


class DOPRI45Controller(StepSizeController):
    """
    Step size control for the DOPRI45 method. The step size is regulated by computing a local
    error estimate from two different-order solutions.
    """

    def __init__(self,
                 atol: float = 0.001,
                 rtol: float = 0.001,
                 fac_min: float = 0.2,
                 fac_max: float = 5.0,
                 safety_factor: float = 0.9):
        """
        DOPRI45 step size control constructor.

        Args:
            atol: Absolute error tolerance in the error estimate.
            rtol: Relative error tolerance in the error estimate.
            fac_min: Maximal step size reduction factor.
            fac_max: Maximal step size increase factor.
            safety_factor: Safety factor, commonly set around 0.9.
        """

        self.atol = atol
        self.rtol = rtol
        self.fac_min = fac_min
        self.fac_max = fac_max
        self.safety_factor = safety_factor
        self.order = 5

    def __call__(self,
                 i: int,
                 h: float,
                 state: ModelState,
                 updated_state: ModelState,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> Tuple[bool, float]:
        """
        DOPRI45 step size control call operator.

        Args:
            i: Current iteration number.
            h: Current step size.
            state: Previous ODE state.
            updated_state: New computed ODE state.
            model: The ODE model being integrated.
            local_vars: Handle for locals() dict passed to the step size control.

        Returns:
            A tuple (acc, h_new) consisting of a boolean acc, indicating whether or not the new
            state was accepted, and the step size h_new to use in the next step.

        """

        order4, order5 = updated_state

        y_prev = state[-1]

        y_4, y_5 = order4[-1], order5[-1]

        err_tol = self.atol + self.rtol * np.maximum(np.abs(y_prev), np.abs(y_5))

        err_ratio = np.linalg.norm((y_4 - y_5) / err_tol)

        accept = err_ratio < 1.

        error_est = (1 / err_ratio) ** (1 / self.order)

        h_new = h * min(self.fac_max, max(self.fac_min, self.safety_factor * error_est))

        return accept, h_new
