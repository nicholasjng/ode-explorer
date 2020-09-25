import numpy as np
import os
import absl
import datetime
import copy

import logging

import absl.logging

from tqdm import tqdm
from typing import Dict, Callable, Text, Any, List
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.model import ODEModel
from utils.data_utils import write_to_file


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
                 data_output_dir: Text = None):

        # step function used to integrate a model
        self.step_func = step_func

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # data container for the steps
        self.result_data = []

        # step count, can be used to track integration runs
        self._step_count = 0

        # empty list holding the metric data
        self.metric_data = []

        # callbacks to be executed after the step
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

        if metrics is None:
            self.metrics = []
        else:
            self.metrics = metrics

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logger = None

        self.set_up_logger(log_dir=self.log_dir)

        self.data_dir = data_output_dir or os.path.join(os.getcwd(), "results")

    def _reset_step_counter(self):
        self._step_count = 0

    def add_callbacks(self, callback_list: List[Callable]):
        self.callbacks = self.callbacks + callback_list

    def add_metrics(self, metric_list: List[Callable]):
        self.metrics = self.metrics + metric_list

    def write_data_to_file(self, data_outfile: Text = None):
        data_outfile = data_outfile or "run_" + \
                           datetime.datetime.now().strftime('%Y-%m-%d')

        write_to_file(self.result_data, self.data_dir, data_outfile)

    def set_up_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.logger = logging.getLogger('Integrator')
        self.logger.info('Creating an Integrator instance.')

    # TODO: Substitute start, y0 -> initial_state ()
    # or subclass Integrator and override the integrate_const method
    def integrate_const(self,
                        model: ODEModel,
                        initial_state: Dict[Text, Any],
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset_step_counter: bool = True,
                        logfile_name: Text = None,
                        verbosity: int = 0,
                        data_outfile: Text = None,
                        flush_data_every: int = None,
                        progress_bar: bool = True):

        # create file handler which logs even debug messages
        logfile = logfile_name or "logs.txt"
        fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
        fh.setLevel(verbosity)
        # create formatter and add it to the handlers
        formatter = absl.logging.PythonFormatter()
        fh.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)

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
            logging.warning("No step size argument was supplied. The step "
                            "size will be set according to the start, end "
                            "and num_steps arguments. This can have a "
                            "negative affect on accuracy.")
        elif not num_steps:
            num_steps = int((end - start) / h)

        # deepcopy here, otherwise the initial state gets overwritten
        state_dict = copy.deepcopy(initial_state)
        self.result_data.append(state_dict)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, num_steps + 2)

        if progress_bar:
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
                           datetime.datetime.now().strftime('%Y-%m-%d')

            self.write_data_to_file(data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(os.path.join(
                self.log_dir, outfile_name)))

        # return self to allow daisy chaining a visualization method
        # TODO: Implement matplotlib based visualization method
        return self
