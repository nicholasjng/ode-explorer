import pandas as pd
import numpy as np

from absl import logging
from tqdm import tqdm
from typing import Dict, Callable, Text, Any, List
from ode_explorer.stepfunction import StepFunction
from ode_explorer.model import ODEModel


class Integrator:
    """
    Base class for all ODE integrators.
    """
    def __init__(self,
                 step_func: StepFunction,
                 pre_step_hook: Callable = None,
                 post_step_callbacks: List[Callable] = None):

        # step function used to integrate a model
        self.step_func = step_func

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # data container for the steps
        self.data = []

        # step count, can be used to track integration runs
        self._step_count = 0

        # callbacks to be executed after the step
        if post_step_callbacks is None:
            self.callbacks = []

    def _reset_step_counter(self):
        self._step_count = 0

    def _increase_step_counter(self):
        self._step_count += 1

    def update_data(self, state_dict: Dict[Text, float]):
        self.data.append(state_dict)

    def integrate_const(self,
                        model: ODEModel,
                        start: float,
                        y0: np.ndarray,
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset_step_counter: bool = True,
                        ) -> None:

        # arg checks for time stepping
        stepping_data = [bool(end), bool(h), bool(num_steps)]

        if not start:
            raise ValueError("A float value has to be given for the "
                             "\"start\" variable.")

        if stepping_data.count(True) != 2:
            raise ValueError("Error: This Integrator run is mis-configured. "
                             "You should specify exactly two of the "
                             "arguments \"end\", \"h\" and \"num_steps\".")

        if reset_step_counter:
            self._reset_step_counter()

        # Register the missing of the 4 arguments
        if not end:
            end = start + h * num_steps
        # TODO: What happens if a step size controller is used?
        elif not h:
            h = (start - end) / num_steps
            logging.warn("No step size argument was supplied. The "
                         "step size will be set according to the start, end "
                         "and num_steps arguments. This can have a negative "
                         "affect on accuracy.")
        elif not num_steps:
            num_steps = (start - end) / h

        state_dict = {**{model.indep_name: start},
                      **dict(zip(model.variable_names, y0))}

        self.update_data(state_dict=state_dict)

        for i in tqdm(range(num_steps + 1)):

            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state_dict = self.step_func.forward(model, state_dict)

            self.update_data(updated_state_dict)

            # execute the registered callbacks after the step
            for callback in self.callbacks:
                callback(self, locals())

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state_dict = updated_state_dict

            self._increase_step_counter()
