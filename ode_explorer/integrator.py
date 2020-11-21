import pandas as pd
import os
import datetime
import copy
import logging
import numpy as np
import uuid
# import matplotlib.pyplot as plt

from tqdm import trange
from tabulate import tabulate
from typing import Dict, Callable, Text, List, Union, Any

from ode_explorer.templates import StepFunction
from ode_explorer.model import ODEModel
from ode_explorer.callbacks import Callback
from ode_explorer.metrics import Metric
from ode_explorer.constants import DataFormatKeys, RunKeys, RunMetadataKeys, RunConfigKeys
from ode_explorer.stepsizecontroller import StepsizeController
from ode_explorer.integrator_loops import constant_h_loop, dynamic_h_loop
from ode_explorer.utils.data_utils import write_to_file, convert_to_zipped

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

        # empty list holding the different executed ODE integration runs
        self.runs = []

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
                 sc: StepsizeController,
                 initial_state: Dict[Text, Union[np.ndarray, float]],
                 end: float,
                 h: float,
                 num_steps: int,
                 reset: bool,
                 verbosity: int,
                 logfile: Text,
                 callbacks: List[Callback],
                 metrics: List[Metric]):

        run.update({RunKeys.RESULT_DATA: [], RunKeys.METRICS: []})

        run_metadata = {RunMetadataKeys.METRIC_NAMES:
                        [m.__name__ for m in metrics],
                        RunMetadataKeys.CALLBACK_NAMES:
                        [c.__name__ for c in callbacks],
                        RunMetadataKeys.TIMESTAMP: datetime.datetime.now(),
                        RunMetadataKeys.RUN_ID: uuid.uuid4(),
                        RunMetadataKeys.STEPFUNC_OUTPUT_FORMAT: step_func.output_format}

        # create file handler
        # TODO: Flush all previous handlers except the base to prevent clutter
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            self.logger.addHandler(fh)
            fh.setLevel(verbosity)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        run_metadata.update({RunMetadataKeys.MODEL_METADATA: model.get_metadata()})

        for handler in self.logger.handlers:
            handler.setLevel(verbosity)

        start = initial_state[model.indep_name]

        run_config = {RunConfigKeys.START: start,
                      RunConfigKeys.END: end,
                      RunConfigKeys.STEP_SIZE: h,
                      RunConfigKeys.NUM_STEPS: num_steps}

        run[RunKeys.RUN_CONFIG] = run_config

        if reset:
            self._reset()

        # append the initial state
        run[RunKeys.RESULT_DATA].append(initial_state)

        initial_metrics = {}

        for metric in metrics:
            val = metric(0, initial_state, initial_state, model, locals())
            initial_metrics[metric.__name__] = val

        if sc:
            # means dynamical integration, hence we log step size, accepts and rejects
            initial_metrics.update({"step_size": h,
                                    "n_accept": 0,
                                    "n_reject": 0})

        run[RunKeys.METRICS].append(initial_metrics)

    def integrate_const(self,
                        model: ODEModel,
                        step_func: StepFunction,
                        initial_state: Dict[Text, Union[np.ndarray, float]],
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset: bool = True,
                        verbosity: int = 0,
                        data_outfile: Text = None,
                        logfile: Text = None,
                        callbacks: List[Callback] = None,
                        metrics: List[Metric] = None):

        # construct run object, dict for now
        run = {}

        # callbacks and metrics, to be executed/computed after the step
        callbacks = callbacks or []
        metrics = metrics or []

        self._startup(run=run,
                      model=model,
                      step_func=step_func,
                      sc=None,
                      initial_state=initial_state,
                      end=end,
                      h=h,
                      num_steps=num_steps,
                      reset=reset,
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
                        metrics=metrics,
                        logger=self.logger)

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
                      sc=sc,
                      initial_state=initial_state,
                      end=end,
                      h=initial_h,
                      num_steps=max_steps,
                      reset=reset,
                      verbosity=verbosity,
                      logfile=logfile,
                      callbacks=callbacks,
                      metrics=metrics)

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)

        self.logger.info("Starting integration.")

        # treat initial state as state 0
        iterator = range(1, max_steps + 1)

        if self.progress_bar:
            # register to tqdm
            iterator = trange(1, max_steps + 1)

        dynamic_h_loop(run=run,
                       iterator=iterator,
                       step_func=step_func,
                       model=model,
                       h=initial_h,
                       state=state,
                       callbacks=callbacks,
                       metrics=metrics,
                       sc=sc,
                       logger=self.logger)

        self.logger.info("Finished integration.")

        if data_outfile:
            self.write_data_to_file(model=model, data_outfile=data_outfile)

            self.logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, data_outfile)))

        self.runs.append(run)

        return self

    def list_runs(self, tablefmt: Text = "github"):
        metadata_list = [run[RunKeys.RUN_METADATA] for run in self.runs]

        print(tabulate(metadata_list, tablefmt=tablefmt))

    def get_run_by_id(self, run_id: Text):
        try:
            run = next(r for r in self.runs if r[RunKeys.RUN_METADATA][RunMetadataKeys.RUN_ID].startswith(run_id))
        except StopIteration:
            raise ValueError(f"Error: Run with ID {run_id} not found.")

        return run

    # TODO: Save some type of run metadata to avoid having to pass
    #  the whole model object
    def return_result_data(self, run_id: Text) -> pd.DataFrame:
        run = self.get_run_by_id(run_id=run_id)

        # TODO: This is ugly
        output_format = run[RunKeys.RUN_METADATA][RunMetadataKeys.STEPFUNC_OUTPUT_FORMAT]
        model_metadata = run[RunKeys.RUN_METADATA][RunMetadataKeys.MODEL_METADATA]

        run_result = copy.deepcopy(run[RunKeys.RESULT_DATA])
        if output_format != DataFormatKeys.ZIPPED:
            for i, res in enumerate(run_result):
                run_result[i] = convert_to_zipped(res, model_metadata=model_metadata)

        return pd.DataFrame(run_result)

    def return_metrics(self, run_id: Text) -> pd.DataFrame:
        run = self.get_run_by_id(run_id=run_id)

        return pd.DataFrame(run[RunKeys.METRICS])

    def visualize(self, run_id: Text, ax=None):

        df = self.return_result_data(run_id=run_id)

        df.plot(ax=ax)
