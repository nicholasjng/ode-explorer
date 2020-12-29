import logging
from typing import Any, List, Dict, Text

from tqdm import trange

from ode_explorer import defaults
from ode_explorer.callbacks import Callback
from ode_explorer.constants import RunKeys, RunConfigKeys
from ode_explorer.metrics import Metric
from ode_explorer.models import BaseModel
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.stepsize_control import StepSizeController
from ode_explorer.types import ModelState

__all__ = ["constant_h_loop", "adaptive_h_loop"]

logger = logging.getLogger(__name__)


def constant_h_loop(run: Dict[Text, Any],
                    step_func: StepFunction,
                    model: BaseModel,
                    h: float,
                    max_steps: int,
                    state: ModelState,
                    callbacks: List[Callback],
                    metrics: List[Metric],
                    progress_bar: bool = False,
                    sc: StepSizeController = None):
    # callbacks and metrics
    callbacks = callbacks or []
    metrics = metrics or []

    run_config = run[RunKeys.RUN_CONFIG]

    validate_const_h_loop(run_config=run_config)

    # treat initial state as state 0
    if progress_bar:
        # register to tqdm
        iterator = trange(1, max_steps + 1)
    else:
        iterator = range(1, max_steps + 1)

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
        state = updated_state


def adaptive_h_loop(run: Dict[Text, Any],
                    step_func: StepFunction,
                    model: BaseModel,
                    h: float,
                    max_steps: int,
                    state: ModelState,
                    callbacks: List[Callback],
                    metrics: List[Metric],
                    sc: StepSizeController = None,
                    progress_bar: bool = False):
    # callbacks and metrics
    callbacks = callbacks or []
    metrics = metrics or []

    run_config = run[RunKeys.RUN_CONFIG]

    validate_dynamic_loop(run_config=run_config)

    max_steps = run_config[RunConfigKeys.NUM_STEPS]

    end = run_config[RunConfigKeys.END]

    # treat initial state as state 0
    if progress_bar:
        # register to tqdm
        iterator = trange(1, max_steps + 1)
    else:
        iterator = range(1, max_steps + 1)

    for i in iterator:
        # if self._pre_step_hook:
        #     self._pre_step_hook()

        updated_state = step_func.forward(model, state, h)

        accepted, h = sc(i, h, state, updated_state, model, locals())

        # e.g. DOPRI45 returns a tuple of estimates, as do embedded RKs
        if isinstance(updated_state, (tuple, list)):
            # TODO: This needs work, maybe infer which one is the higher order
            lower_order_sol, higher_order_sol = updated_state
            current = higher_order_sol[0]
        else:
            higher_order_sol = updated_state
            current = higher_order_sol[0]

        if current + h > end:
            h = end - current

        # initialize with the current iteration number and time stamp
        new_metrics = {defaults.iteration: i,
                       defaults.step_size: h,
                       defaults.accepted: int(accepted),
                       defaults.rejected: int(not accepted)}

        for metric in metrics:
            new_metrics[metric.__name__] = metric(i, state, higher_order_sol, model, locals())

        run[RunKeys.METRICS].append(new_metrics)

        # execute the registered callbacks after the step
        for callback in callbacks:
            callback(i, state, higher_order_sol, model, locals())

        if not accepted:
            continue

        run[RunKeys.RESULT_DATA].append(higher_order_sol)

        if current >= end:
            break

        # update delayed after callback execution so that callbacks have
        # access to both the previous and the current state
        state = higher_order_sol


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
        logger.warning("No step size was supplied. The step size will be "
                       "set according to the given start, end "
                       "and num_steps values. This can potentially have "
                       "a negative affect on accuracy.")
    elif not num_steps:
        num_steps = int((end - start) / h)
        run_config.update({RunConfigKeys.NUM_STEPS: num_steps})


def validate_dynamic_loop(run_config: Dict[Text, Any]):
    start = run_config[RunConfigKeys.START]
    end = run_config[RunConfigKeys.END]
    max_steps = run_config[RunConfigKeys.NUM_STEPS]
    initial_h = run_config[RunConfigKeys.STEP_SIZE]

    if start > end:
        raise ValueError("The upper integration bound has to be larger "
                         "than the starting value.")

    if not initial_h:
        logger.warning(f"No initial step size supplied, falling "
                       f"back to builtin initial step size "
                       f"of {defaults.INITIAL_H}.")
        initial_h = defaults.INITIAL_H
        run_config[RunConfigKeys.STEP_SIZE] = initial_h

    if not max_steps:
        logger.warning(f"No maximum step count supplied, falling "
                       f"back to builtin maximum step count "
                       f"of {defaults.MAX_STEPS}.")
        max_steps = defaults.MAX_STEPS
        run_config[RunConfigKeys.NUM_STEPS] = max_steps
