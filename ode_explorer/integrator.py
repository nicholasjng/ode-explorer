import pandas as pd
import os
import datetime
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from typing import Dict, Callable, Text, List, Union

from ode_explorer.templates import StepFunction
from ode_explorer.model import ODEModel
from ode_explorer.callbacks import Callback
from ode_explorer.metrics import Metric
from ode_explorer.constants import DYNAMIC_MAX_STEPS, DYNAMIC_INITIAL_H
from ode_explorer.stepsizecontroller import StepsizeController

from utils.data_utils import write_to_file, convert_to_zipped
from utils.data_utils import infer_dict_format

logging.basicConfig(level=logging.DEBUG)
integrator_logger = logging.getLogger("ode_explorer.integrator.Integrator")


class Integrator:
    """
    Base class for all ODE integrators.
    """
    def __init__(self,
                 pre_step_hook: Callable = None,
                 log_dir: Text = None,
                 logfile_name: Text = None,
                 data_output_dir: Text = None,
                 progress_bar: bool = True):

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # empty lists holding the step and metric data
        self.result_data, self.metric_data = [], []

        # step count, can be used to track integration runs
        self._step_count = 0

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self.logger = None

        self.set_up_logger(log_dir=self.log_dir)

        self.data_dir = data_output_dir or os.path.join(os.getcwd(), "results")

        self.progress_bar = progress_bar

    def _reset_step_counter(self):
        self._step_count = 0

    def write_data_to_file(self, model, data_outfile: Text = None):
        data_outfile = data_outfile or "run_" + \
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        write_to_file(self.result_data, model, self.data_dir, data_outfile)

    def set_up_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.logger = integrator_logger
        # flush handlers on construction since it is a global object
        self.logger.handlers = []

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, self.logfile_name))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.info('Creating an Integrator instance.')

    def integrate_const(self,
                        model: ODEModel,
                        step_func: StepFunction,
                        initial_state: Dict[Text, Union[np.ndarray, float]],
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset_step_counter: bool = True,
                        verbosity: int = 0,
                        data_outfile: Text = None,
                        logfile: Text = None,
                        flush_data_every: int = None,
                        callbacks: List[Callback] = None,
                        metrics: List[Metric] = None):

        # callbacks and metrics, to be executed/computed after the step
        callbacks = callbacks or []
        metrics = metrics or []

        # create file handler
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            self.logger.addHandler(fh)
            fh.setLevel(verbosity)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        input_format = infer_dict_format(state_dict=initial_state, model=model)

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        # arg checks for time stepping
        stepping_data = [bool(end), bool(h), bool(num_steps)]

        start = initial_state[model.indep_name]

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

        if reset_step_counter:
            self._reset_step_counter()

        # Register the missing of the 4 arguments
        if not end:
            end = start + h * num_steps
        elif not h:
            h = (end - start) / num_steps
            integrator_logger.warning(
                            "No step size was supplied. The step size will be "
                            "set according to the given start, end "
                            "and num_steps values. This can potentially have "
                            "a negative affect on accuracy.")
        elif not num_steps:
            num_steps = int((end - start) / h)

        if not flush_data_every:
            flush_data_every = num_steps + 2

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)
        self.result_data.append(initial_state)

        metric_dict = {"iteration": 0,
                       "step_size": h}

        # TODO: This can very well break
        metric_dict.update({m.name: 0.0 for m in metrics})
        self.metric_data.append(metric_dict)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, num_steps + 2)

        if self.progress_bar:
            # register to tqdm
            iterator = trange(1, num_steps + 2)

        for i in iterator:
            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state = step_func.forward(model, state, h)

            self.result_data.append(updated_state)

            if i % flush_data_every == 0:
                self.write_data_to_file(data_outfile)
                self.result_data = []

            # execute the registered callbacks after the step
            for callback in callbacks:
                callback(i, state, updated_state, model, locals())

            # adding the current iteration number and time stamp
            metric_dict = {"iteration": i,
                           "step_size": h}

            for metric in metrics:
                val = metric(i, state, updated_state, model, locals())
                metric_dict[metric.__name__] = val

            self.metric_data.append(metric_dict)

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state.update(updated_state)

            self._step_count += 1

        self.logger.info("Finished integration.")

        if self.result_data:
            outfile_name = data_outfile or "run_" + \
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            self.write_data_to_file(model=model, data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, outfile_name)))

        return self

    def integrate_dynamically(self,
                              model: ODEModel,
                              step_func: StepFunction,
                              sc: Union[StepsizeController, Callable],
                              initial_state: Dict[Text,
                                                  Union[np.ndarray, float]],
                              end: float,
                              initial_h: float = None,
                              max_steps: int = None,
                              reset_step_counter: bool = True,
                              verbosity: int = logging.INFO,
                              data_outfile: Text = None,
                              logfile: Text = None,
                              flush_data_every: int = None,
                              callbacks: List[Callback] = None,
                              metrics: List[Metric] = None):

        # callbacks and metrics, to be executed/computed after the step
        callbacks = callbacks or []
        metrics = metrics or []

        # create file handlers if necessary
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            fh.setLevel(verbosity)
            self.logger.addHandler(fh)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        input_format = infer_dict_format(state_dict=initial_state, model=model)

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        if reset_step_counter:
            self._reset_step_counter()

        start = float(initial_state[model.indep_name])

        if start > end:
            raise ValueError("The upper integration bound has to be larger "
                             "than the starting value.")

        if not initial_h:
            self.logger.warning(f"No maximum step count supplied, falling "
                                f"back to builtin initial step size "
                                f"of {DYNAMIC_INITIAL_H}.")
            max_steps = DYNAMIC_INITIAL_H

        if not max_steps:
            self.logger.warning(f"No maximum step count supplied, falling "
                                f"back to builtin maximum step count "
                                f"of {DYNAMIC_MAX_STEPS}.")
            max_steps = DYNAMIC_MAX_STEPS

        flush_data_every = flush_data_every or max_steps + 2

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)
        self.result_data.append(initial_state)

        initial_metrics = {"iteration": 0,
                           "step_size": initial_h,
                           "n_accept": 0,
                           "n_reject": 0}

        # TODO: This can very well break
        initial_metrics.update({m.name: 0.0 for m in metrics})
        self.metric_data.append(initial_metrics)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, max_steps + 2)

        if self.progress_bar:
            # register to tqdm
            iterator = trange(1, max_steps + 2)

        h = initial_h

        for i in iterator:
            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state = step_func.forward(model, state, h)

            self.result_data.append(updated_state)

            if i % flush_data_every == 0:
                self.write_data_to_file(data_outfile)
                self.result_data = []

            # execute the registered callbacks after the step
            for callback in callbacks:
                callback(i, state, updated_state, model, locals())

            accepted, h_new = sc(i, h, state, updated_state, model, locals())
            h = h_new

            # initialize with the current iteration number and time stamp
            new_metrics = {"iteration": i,
                           "step_size": h,
                           "n_accept": self.metric_data[i - 1]["n_accept"] + int(accepted),
                           "n_reject": self.metric_data[i - 1]["n_reject"] + int(not accepted)}

            if not accepted:
                continue



            for metric in metrics:
                new_metrics[metric.__name__] = metric(i, state, updated_state, model, locals())

            self.metric_data.append(new_metrics)

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state.update(updated_state)

            self._step_count += 1

        self.logger.info("Finished integration.")

        if self.result_data:
            outfile_name = data_outfile or "run_" + \
                           datetime.datetime.now().strftime(
                               '%Y-%m-%d-%H-%M-%S')

            self.write_data_to_file(model=model, data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, outfile_name)))

        return self

    def visualize(self, model: ODEModel, ax=None):

        for i, res in enumerate(self.result_data):
            if not all(key in res for key in model.dim_names):
                self.result_data[i] = convert_to_zipped(res, model)

        df = pd.DataFrame(self.result_data)

        df.plot(ax=ax)
