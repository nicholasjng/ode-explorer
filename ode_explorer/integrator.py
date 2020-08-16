import numpy as np
import os

from absl import logging
from tqdm import tqdm
from typing import Dict, Callable, Text, Any, List
from ode_explorer.stepfunction import StepFunction
from ode_explorer.model import ODEModel
from utils.data_utils import write_results_to_file


class Integrator:
    """
    Base class for all ODE integrators.
    """
    def __init__(self,
                 step_func: StepFunction,
                 pre_step_hook: Callable = None,
                 post_step_callbacks: List[Callable] = None,
                 log_dir: Text = None):

        # step function used to integrate a model
        self.step_func = step_func

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # data container for the steps
        self.result_data = []

        # step count, can be used to track integration runs
        self._step_count = 0

        # callbacks to be executed after the step
        if post_step_callbacks is None:
            self.callbacks = []

        self.log_dir = log_dir or os.getcwd()

    def _reset_step_counter(self):
        self._step_count = 0

    def add_callbacks(self, callback_list: List[Callable]):
        self.callbacks = self.callbacks + callback_list

    def integrate_const(self,
                        model: ODEModel,
                        start: float,
                        y0: np.ndarray,
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset_step_counter: bool = True,
                        logfile_name: Text = None,
                        **kwargs) -> None:

        # arg checks for time stepping
        stepping_data = [bool(end), bool(h), bool(num_steps)]

        if not start:
            raise ValueError("A float value has to be given for the "
                             "\"start\" variable.")

        if end and (start > end):
            raise ValueError("The upper integration bound has to be larger "
                             "than the starting value.")

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
            h = (end - start) / num_steps
            logging.warning("No step size argument was supplied. The step "
                            "size will be set according to the start, end "
                            "and num_steps arguments. This can have a "
                            "negative affect on accuracy.")
        elif not num_steps:
            num_steps = int((end - start) / h)

        state_dict = {**{model.indep_name: start},
                      **dict(zip(model.variable_names, y0))}

        self.result_data.append(state_dict)

        for i in tqdm(range(num_steps + 1)):

            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state_dict = self.step_func.forward(model, state_dict,
                                                        h, **kwargs)

            self.result_data.append(updated_state_dict)

            # execute the registered callbacks after the step
            # scikit-learn inspired callback signature
            for callback in self.callbacks:
                callback(self, model, locals())

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state_dict = updated_state_dict

            self._step_count += 1

        if self.result_data:

            outfile_name = logfile_name or "run_" + \
                           datetime.datetime.now().strftime('%Y-%m-%d')

            write_results_to_file(self.result_data, out_dir=self.log_dir,
                                  out_name=outfile_name)
