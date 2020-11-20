import pandas as pd
import os
import datetime
import copy
import logging
import numpy as np
#import matplotlib.pyplot as plt

from tqdm import trange
from typing import Dict, Callable, Text, List, Union, Any

from ode_explorer.templates import StepFunction
from ode_explorer.model import ODEModel
from ode_explorer.callbacks import Callback
from ode_explorer.metrics import Metric
from ode_explorer.constants import DataFormatKeys, DynamicVariables, \
    RunKeys, RunMetadataKeys, RunConfigKeys
from ode_explorer.stepsizecontroller import StepsizeController
from ode_explorer.integrator_loops import constant_h_loop

from ode_explorer.utils.data_utils import write_to_file, convert_to_zipped
from ode_explorer.utils.data_utils import infer_dict_format

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
        self.runs = []
        self.result_data, self.metric_data = [], []

        # step count, can be used to track integration runs
        self._step_count = 0

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self.logger = integrator_logger

        self._set_up_logger(log_dir=self.log_dir)

        self.data_dir = data_output_dir or os.path.join(os.getcwd(), "results")

        self.progress_bar = progress_bar

        self.logger.info("Created an Integrator instance.")

    def _reset(self):
        # Hard reset all data and step counts
        self._step_count = 0
        self.runs = []

    def write_data_to_file(self, model, data_outfile: Text = None):
        data_outfile = data_outfile or "run_" + \
                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        write_to_file(self.result_data, model, self.data_dir, data_outfile)

    def _set_up_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # flush handlers on construction since it is a global object
        self.logger.handlers = []

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.log_dir, self.logfile_name))
        fh.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.info('Creating an Integrator instance.')

    def _startup(self,
                 run: Dict[Text, Any],
                 model: ODEModel,
                 step_func: StepFunction,
                 initial_state: Dict[Text, Union[np.ndarray, float]],
                 end: float,
                 h: float,
                 num_steps: int,
                 reset_step_counter: bool,
                 verbosity: int,
                 logfile: Text,
                 callbacks: List[Callback],
                 metrics: List[Metric]):

        run.update({RunKeys.RESULT_DATA: [], RunKeys.METRICS: []})

        run_metadata = {RunMetadataKeys.METRIC_NAMES:
                        [m.__name__ for m in metrics],
                        RunMetadataKeys.CALLBACK_NAMES:
                        [c.__name__ for c in callbacks],
                        RunMetadataKeys.TIMESTAMP: datetime.datetime.now()}

        # create file handler
        # TODO: Flush all previous handlers except the base to prevent clutter
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            self.logger.addHandler(fh)
            fh.setLevel(verbosity)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        run_metadata.update({RunMetadataKeys.DIM_NAMES: model.dim_names,
                             RunMetadataKeys.VARIABLE_NAMES: model.variable_names,
                             RunMetadataKeys.STEPFUNC_OUTPUT_FORMAT: step_func.output_format})

        input_format = infer_dict_format(state_dict=initial_state, model=model)

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        start = initial_state[model.indep_name]

        run_config = {RunConfigKeys.START: start,
                      RunConfigKeys.END: end,
                      RunConfigKeys.STEP_SIZE: h,
                      RunConfigKeys.NUM_STEPS: num_steps}

        run[RunKeys.RUN_CONFIG] = run_config

        if reset_step_counter:
            self._reset()

        # append the initial state
        run[RunKeys.RESULT_DATA].append(initial_state)

        metric_dict = {}

        for metric in metrics:
            val = metric(0, initial_state, initial_state, model, locals())
            metric_dict[metric.__name__] = val
        run[RunKeys.METRICS].append(metric_dict)

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

        # construct run object
        run = {}

        # callbacks and metrics, to be executed/computed after the step
        callbacks = callbacks or []
        metrics = metrics or []

        self._startup(run=run,
                      model=model,
                      step_func=step_func,
                      initial_state=initial_state,
                      end=end,
                      h=h,
                      num_steps=num_steps,
                      reset_step_counter=reset_step_counter,
                      verbosity=verbosity,
                      logfile=logfile,
                      callbacks=callbacks,
                      metrics=metrics)

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)

        self.logger.info("Starting integration.")

        if self.progress_bar:
            # register to tqdm
            iterator = trange(1, num_steps + 1)
        else:
            # treat initial state as state 0
            iterator = range(1, num_steps + 1)

        constant_h_loop(run=run,
                        iterator=iterator,
                        step_func=step_func,
                        model=model,
                        h=h,
                        state=state,
                        callbacks=callbacks,
                        metrics=metrics)

        self.logger.info("Finished integration.")

        if data_outfile:
            self.write_data_to_file(model=model, data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, data_outfile)))

        self.runs.append(run)

        return self

    def integrate_dynamically(self,
                              model: ODEModel,
                              step_func: StepFunction,
                              initial_state: Dict[Text, Union[np.ndarray, float]],
                              sc: Union[StepsizeController, Callable],
                              end: float,
                              initial_h: float = None,
                              max_steps: int = None,
                              reset: bool = True,
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
        # TODO: Flush all previous handlers except the base to prevent clutter
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            fh.setLevel(verbosity)
            self.logger.addHandler(fh)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        input_format = infer_dict_format(state_dict=initial_state, model=model)

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        if reset:
            self._reset()

        start = float(initial_state[model.indep_name])

        if start > end:
            raise ValueError("The upper integration bound has to be larger "
                             "than the starting value.")

        if not initial_h:
            self.logger.warning(f"No maximum step count supplied, falling "
                                f"back to builtin initial step size "
                                f"of {DynamicVariables.INITIAL_H}.")
            initial_h = DynamicVariables.INITIAL_H

        if not max_steps:
            self.logger.warning(f"No maximum step count supplied, falling "
                                f"back to builtin maximum step count "
                                f"of {DynamicVariables.MAX_STEPS}.")
            max_steps = DynamicVariables.MAX_STEPS

        flush_data_every = flush_data_every or max_steps + 1

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)
        self.result_data.append(initial_state)

        initial_metrics = {"iteration": 0,
                           "step_size": initial_h,
                           "n_accept": 0,
                           "n_reject": 0}

        # TODO: This can very well break
        initial_metrics.update({m.__name__: 0.0 for m in metrics})
        self.metric_data.append(initial_metrics)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, max_steps + 1)

        if self.progress_bar:
            # register to tqdm
            iterator = trange(1, max_steps + 1)

        h = initial_h

        for i in iterator:
            if self._pre_step_hook:
                self._pre_step_hook()

            updated_state = step_func.forward(model, state, h)

            accepted, h = sc(i, h, state, updated_state, model, locals())

            # e.g. DOPRI45 returns a tuple of estimates, as do embedded RKs
            if isinstance(updated_state, (tuple, list)):
                current = updated_state[0][model.indep_name]
            else:
                current = updated_state[model.indep_name]

            if current + h > end:
                h = end - current

            # initialize with the current iteration number and time stamp
            new_metrics = {"iteration": i,
                           "step_size": h,
                           "n_accept": self.metric_data[i - 1]["n_accept"] + int(accepted),
                           "n_reject": self.metric_data[i - 1]["n_reject"] + int(not accepted)}

            for metric in metrics:
                new_metrics[metric.__name__] = metric(i, state, updated_state, model, locals())
                self.metric_data.append(new_metrics)

            # execute the registered callbacks after the step
            for callback in callbacks:
                callback(i, state, updated_state, model, locals())

            if not accepted:
                continue

            self.result_data.append(updated_state)

            if current >= end:
                break

            if len(self.result_data) % flush_data_every == 0:
                if data_outfile:
                    self.write_data_to_file(data_outfile)
                    self.result_data = []

            # update delayed after callback execution so that callbacks have
            # access to both the previous and the current state
            state.update(updated_state)

            self._step_count += 1

        self.logger.info("Finished integration.")

        if self.result_data:
            if data_outfile:
                self.write_data_to_file(model=model, data_outfile=data_outfile)

                self.logger.info("Results written to file {}.".format(
                    os.path.join(self.log_dir, data_outfile)))

        return self

    # TODO: Save some type of run metadata to avoid having to pass
    #  the whole model object
    def return_result_data(self, model: ODEModel) -> pd.DataFrame:
        first_step = self.result_data[1]

        result_format = infer_dict_format(first_step, model=model)

        if result_format != DataFormatKeys.ZIPPED:
            for i, res in enumerate(self.result_data):
                self.result_data[i] = convert_to_zipped(res, model)

        return pd.DataFrame(self.result_data)

    def return_metrics(self, model: ODEModel) -> pd.DataFrame:

        return pd.DataFrame(self.metric_data)

    def visualize(self, model: ODEModel, ax=None):

        df = self.return_result_data(model=model)

        df.plot(ax=ax)