import copy
import logging
from typing import Any, List

from tqdm import trange

from ode_explorer.callbacks import Callback
from ode_explorer.metrics import Metric
from ode_explorer.models import BaseModel
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.stepsize_control import StepSizeController
from ode_explorer.types import State

__all__ = ["constant_h_loop", "adaptive_h_loop"]

progress_funcs = {True: trange, False: range}

logger = logging.getLogger(__name__)


def constant_h_loop(step_func: StepFunction,
                    model: BaseModel,
                    h: Any,
                    max_steps: int,
                    end: float,
                    initial_state: State,
                    callbacks: List[Callback],
                    metrics: List[Metric],
                    progress_bar: bool = False,
                    sc: StepSizeController = None):
    result = []

    # deepcopy here, otherwise the initial state gets overwritten
    state = copy.deepcopy(initial_state)

    # treat initial state as state 0
    iterator = progress_funcs.get(bool(progress_bar))(1, max_steps + 1)

    for i in iterator:
        new_state = step_func.forward(model, state, h)

        # adding the current iteration number and time stamp
        new_metrics = {m.__name__: m(i, state, new_state, model, locals()) for m in metrics}

        # execute the registered callbacks after the step
        for callback in callbacks:
            callback(i, state, new_state, model, locals())

        result.append((*new_state, new_metrics))

        # update delayed after callback execution
        state = new_state

    return result


def adaptive_h_loop(step_func: StepFunction,
                    model: BaseModel,
                    h: float,
                    max_steps: int,
                    end: float,
                    initial_state: State,
                    callbacks: List[Callback],
                    metrics: List[Metric],
                    sc: StepSizeController = None,
                    progress_bar: bool = False):
    result = []

    state = copy.deepcopy(initial_state)

    # treat initial state as state 0
    iterator = progress_funcs.get(bool(progress_bar))(1, max_steps + 1)

    for i in iterator:
        new_state = step_func.forward(model, state, h)

        accepted, h = sc(i, h, state, new_state, model, locals())

        # method is assumed to return a tuple of estimates
        current, (sol1, sol2) = new_state

        if current + h > end:
            h = end - current

        # adding the current iteration number and time stamp
        new_metrics = {m.__name__: m(i, state, new_state, model, locals()) for m in metrics}

        # execute the registered callbacks after the step
        for callback in callbacks:
            callback(i, state, new_state, model, locals())

        if not accepted:
            continue

        result.append((current, sol1, new_metrics))

        if current >= end:
            break

        # update delayed after callback execution
        state = (current, sol1)

    return result
