import numpy as np
import logging
from typing import Any, List, Dict, Union, Text
from ode_explorer.templates import StepFunction
from ode_explorer.model import ODEModel
from ode_explorer.callbacks import Callback
from ode_explorer.metrics import Metric
from ode_explorer.stepsizecontroller import StepsizeController
from ode_explorer.constants import RunKeys, RunConfigKeys

__all__ = ["constant_h_loop"]


def constant_h_loop(run: Dict[Text, Any],
                    iterator: Any,
                    step_func: StepFunction,
                    model: ODEModel,
                    h: float,
                    state: Dict[Text, Union[float, np.ndarray]],
                    callbacks: List[Callback],
                    metrics: List[Metric],
                    sc: StepsizeController = None):

    run_config = run[RunKeys.RUN_CONFIG]

    validate_const_h_loop(run_config=run_config)

    for i in iterator:
        # if self._pre_step_hook:
        #     self._pre_step_hook()

        updated_state = step_func.forward(model, state, h)

        # adding the current iteration number and time stamp
        metric_dict = {}

        for metric in metrics:
            val = metric(i, state, updated_state, model, locals())
            metric_dict[metric.__name__] = val

        run[RunKeys.METRICS].append(metric_dict)

        # execute the registered callbacks after the step
        for callback in callbacks:
            callback(i, state, updated_state, model, locals())

        run[RunKeys.RESULT_DATA].append(updated_state)

        # update delayed after callback execution so that callbacks have
        # access to both the previous and the current state
        state.update(updated_state)


def validate_const_h_loop(run_config: Dict[Text, Any]):

    start = run_config[RunConfigKeys.START]
    end = run_config[RunConfigKeys.END]
    num_steps = run_config[RunConfigKeys.NUM_STEPS]
    h = run_config[RunConfigKeys.STEP_SIZE]

    # arg checks for time stepping
    stepping_data = [bool(end), bool(h), bool(num_steps)]

    if not isinstance(start, float):
        raise ValueError("A float value has to be given for the "
                         "\"start\" variable.")

    if end and (start > end):
        raise ValueError("The upper integration bound has to be larger "
                         "than the starting value.")

    if stepping_data.count(True) != 2:
        raise ValueError("Error: This Integrator run is mis-configured. "
                         "You should specify exactly two of the "
                         "arguments \"end\", \"h\" and \"num_steps\".")

    # Register the missing of the 4 arguments
    if not end:
        end = start + h * num_steps
        run_config.update({RunConfigKeys.END: end})
    elif not h:
        h = (end - start) / num_steps
        run_config.update({RunConfigKeys.STEP_SIZE: h})
        logging.warning("No step size was supplied. The step size will be "
                        "set according to the given start, end "
                        "and num_steps values. This can potentially have "
                        "a negative affect on accuracy.")
    elif not num_steps:
        num_steps = int((end - start) / h)
        run_config.update({RunConfigKeys.NUM_STEPS: num_steps})

