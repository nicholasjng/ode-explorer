# import numpy as np
import pandas as pd
import os
import datetime
import copy

# import matplotlib.pyplot as plt

import logging

from tqdm import tqdm
from typing import Dict, Callable, Text, Any, List
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.model import ODEModel
from utils.data_utils import write_to_file, convert_to_zipped

logging.basicConfig(level=logging.DEBUG)
integrator_logger = logging.getLogger("ode_explorer.integrator.Integrator")


class Integrator:
    """
    Base class for all ODE integrators.
    """
    def __init__(self,
                 step_func: StepFunction,
                 pre_step_hook: Callable = None,
                 callbacks: List[Callable] = None,
                 metrics: List[Callable] = None,
                 log_dir: Text = None,
                 logfile_name: Text = None,
                 data_output_dir: Text = None,
                 progress_bar: bool = True):

        # step function used to integrate a model
        self.step_func = step_func

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # empty lists holding the step and metric data
        self.result_data, self.metric_data = [], []

        # step count, can be used to track integration runs
        self._step_count = 0

        # callbacks and metrics, to be executed/computed after the step
        self.callbacks = callbacks or []
        self.metrics = metrics or []

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self.logger = None

        self.setup_logger(log_dir=self.log_dir)

        self.data_dir = data_output_dir or os.path.join(os.getcwd(), "results")

        self.progress_bar = progress_bar

    def _reset_step_counter(self):
        self._step_count = 0

    def add_callbacks(self, callback_list: List[Callable]):
        self.callbacks = self.callbacks + callback_list

    def add_metrics(self, metric_list: List[Callable]):
        self.metrics = self.metrics + metric_list

    def write_data_to_file(self, model, data_outfile: Text = None):
        data_outfile = data_outfile or "run_" + \
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        write_to_file(self.result_data, model, self.data_dir, data_outfile)

    def setup_logger(self, log_dir):
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
                        initial_state: Dict[Text, Any],
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset_step_counter: bool = True,
                        verbosity: int = 0,
                        data_outfile: Text = None,
                        logfile: Text = None,
                        flush_data_every: int = None):

        # create file handler
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            self.logger.addHandler(fh)
            fh.setLevel(verbosity)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        if not flush_data_every:
            flush_data_every = num_steps + 2

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
                            "No step size argument was supplied. The step "
                            "size will be set according to the start, end "
                            "and num_steps arguments. This can have a "
                            "negative affect on accuracy.")
        elif not num_steps:
            num_steps = int((end - start) / h)

        # deepcopy here, otherwise the initial state gets overwritten
        state_dict = copy.deepcopy(initial_state)
        self.result_data.append(initial_state)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, num_steps + 2)

        if self.progress_bar:
            # register to tqdm
            iterator = tqdm(iterator)

        for i in iterator:
            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state_dict = self.step_func.forward(model, state_dict, h)

            self.result_data.append(updated_state_dict)

            if i % flush_data_every == 0:
                self.write_data_to_file(data_outfile)
                self.result_data = []

            # execute the registered callbacks after the step
            for callback in self.callbacks:
                callback(self, model, locals())

            metric_dict = {}
            for metric in self.metrics:
                metric_dict[metric.__name__] = metric(self, model, locals())

            # adding the current time stamp
            metric_dict.update({model.indep_name:
                                updated_state_dict[model.indep_name]})

            self.metric_data.append(metric_dict)

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state_dict.update(updated_state_dict)

            self._step_count += 1

        self.logger.info("Finished integration.")

        if self.result_data:
            outfile_name = data_outfile or "run_" + \
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

            self.write_data_to_file(model=model, data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, outfile_name)))

        return self

    def visualize(self, model: ODEModel, ax=None):

        for i, res in enumerate(self.result_data):
            self.result_data[i] = convert_to_zipped(res, model)

        df = pd.DataFrame(self.result_data)

        df.plot(ax=ax)
