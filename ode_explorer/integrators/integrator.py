import copy
import datetime
import logging
import os
import uuid
from typing import Dict, Callable, Text, List, Union, Any

import absl.logging
import pandas as pd
from tabulate import tabulate
from tqdm import trange

from ode_explorer.callbacks.callback import Callback
from ode_explorer.constants import RunKeys, RunMetadataKeys, RunConfigKeys
from ode_explorer.integrators.integrator_loops import constant_h_loop, dynamic_h_loop
from ode_explorer.metrics.metric import Metric
from ode_explorer.models.model import ODEModel
from ode_explorer.stepfunctions.templates import StepFunction
from ode_explorer.stepsize_control.stepsizecontroller import StepSizeController
from ode_explorer.types import ModelState
from ode_explorer.utils.data_utils import convert_to_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Integrator:
    """
    Base class for all ODE integrators.
    """

    def __init__(self,
                 pre_step_hook: Callable = None,
                 log_dir: Text = None,
                 logfile_name: Text = None,
                 output_dir: Text = None):

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # empty list holding the different executed ODE integration runs
        self.runs = []

        # step count, can be used to track integration runs
        self._step_count = 0

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self._set_up_logger(log_dir=self.log_dir)

        self.output_dir = output_dir or os.path.join(os.getcwd(), "results")

        logger.info("Created an Integrator instance.")

    def _reset(self):
        # Hard reset all data and step counts
        self._step_count = 0
        self.runs = []

    def _set_up_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # flush handlers on construction since it is a global object
        logger.handlers = []

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(absl.logging.PythonFormatter())
        fh = logging.FileHandler(os.path.join(self.log_dir, self.logfile_name))
        fh.setLevel(logging.INFO)
        fh.setFormatter(absl.logging.PythonFormatter())
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.info('Creating an Integrator instance.')

    def _startup(self,
                 run: Dict[Text, Any],
                 model: ODEModel,
                 step_func: StepFunction,
                 sc: Union[StepSizeController, Callable],
                 initial_state: ModelState,
                 end: float,
                 h: float,
                 num_steps: int,
                 reset: bool,
                 verbosity: int,
                 logfile: Text,
                 callbacks: List[Callback],
                 metrics: List[Metric]):

        run_metadata = {RunMetadataKeys.TIMESTAMP: datetime.datetime.now(),
                        RunMetadataKeys.RUN_ID: uuid.uuid4()}

        # create file handler
        # TODO: Flush all previous handlers except the base to prevent clutter
        if logfile:
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            logger.addHandler(fh)
            fh.setLevel(verbosity)

        # initialize dimension names
        model.initialize_dim_names(initial_state)

        for handler in logger.handlers:
            handler.setLevel(verbosity)

        start = initial_state[0]

        run_config = {RunConfigKeys.START: start,
                      RunConfigKeys.END: end,
                      RunConfigKeys.STEP_SIZE: h,
                      RunConfigKeys.NUM_STEPS: num_steps,
                      RunConfigKeys.METRIC_NAMES:
                          [m.__name__ for m in metrics],
                      RunConfigKeys.CALLBACK_NAMES:
                          [c.__name__ for c in callbacks]
                      }

        if reset:
            self._reset()

        initial_metrics = {}

        for metric in metrics:
            val = metric(0, initial_state, initial_state, model, locals())
            initial_metrics[metric.__name__] = val

        if bool(sc):
            # means dynamical integration, hence we log step size, accepts and rejects
            initial_metrics.update({"step_size": h,
                                    "n_accept": 0,
                                    "n_reject": 0})

        run.update({RunKeys.MODEL_METADATA: model.get_metadata(),
                    RunKeys.RUN_METADATA: run_metadata,
                    RunKeys.RUN_CONFIG: run_config,
                    RunKeys.RESULT_DATA: [initial_state],
                    RunKeys.METRICS: [initial_metrics]})

    def integrate_const(self,
                        model: ODEModel,
                        step_func: StepFunction,
                        initial_state: ModelState,
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset: bool = False,
                        verbosity: int = 0,
                        data_outfile: Text = None,
                        logfile: Text = None,
                        callbacks: List[Callback] = None,
                        metrics: List[Metric] = None,
                        progress_bar: bool = False):

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

        logger.info("Starting integration.")

        if progress_bar:
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
                        logger=logger)

        logger.info("Finished integration.")

        if data_outfile:
            self.write_data_to_file(run=run,
                                    model_metadata=model.get_metadata(),
                                    data_outfile=data_outfile)

            logger.info("Results written to file {}.".format(
                os.path.join(self.output_dir, data_outfile)))

        self.runs.append(run)

        return self

    def integrate_dynamically(self,
                              model: ODEModel,
                              step_func: StepFunction,
                              initial_state: ModelState,
                              sc: Union[StepSizeController, Callable],
                              end: float,
                              initial_h: float = None,
                              max_steps: int = None,
                              reset: bool = False,
                              verbosity: int = logging.INFO,
                              data_outfile: Text = None,
                              logfile: Text = None,
                              callbacks: List[Callback] = None,
                              metrics: List[Metric] = None,
                              progress_bar: bool = False):

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

        logger.info("Starting integration.")

        # treat initial state as state 0
        if progress_bar:
            # register to tqdm
            iterator = trange(1, max_steps + 1)
        else:
            iterator = range(1, max_steps + 1)

        dynamic_h_loop(run=run,
                       iterator=iterator,
                       step_func=step_func,
                       model=model,
                       h=initial_h,
                       state=state,
                       callbacks=callbacks,
                       metrics=metrics,
                       sc=sc,
                       logger=logger)

        logger.info("Finished integration.")

        if data_outfile:
            self.write_data_to_file(run=run,
                                    model_metadata=model.get_metadata(),
                                    data_outfile=data_outfile)

            logger.info("Results written to file {}.".format(
                os.path.join(self.log_dir, data_outfile)))

        self.runs.append(run)

        return self

    def list_runs(self, tablefmt: Text = "github"):
        if len(self.runs) == 0:
            print("No runs available!")
            return

        metadata_list = [run[RunKeys.RUN_METADATA] for run in self.runs]

        print(tabulate(metadata_list, headers="keys", tablefmt=tablefmt))

    def get_run_by_id(self, run_id: Text):
        if len(self.runs) == 0:
            raise ValueError("No runs available. Please integrate a model first!")
        if run_id == "latest":
            return self.runs[-1]
        try:
            run = next(r for r in self.runs if run_id in str(r[RunKeys.RUN_METADATA][RunMetadataKeys.RUN_ID]))
        except StopIteration:
            raise ValueError(f"Run with ID {run_id} not found.")

        return run

    def return_result_data(self, run_id: Text) -> pd.DataFrame:
        run = self.get_run_by_id(run_id=run_id)

        model_metadata = run[RunKeys.MODEL_METADATA]

        run_result = copy.deepcopy(run[RunKeys.RESULT_DATA])

        for i, res in enumerate(run_result):
            run_result[i] = convert_to_dict(res, model_metadata=model_metadata)

        return pd.DataFrame(run_result)

    def return_metrics(self, run_id: Text) -> pd.DataFrame:
        run = self.get_run_by_id(run_id=run_id)

        return pd.DataFrame(run[RunKeys.METRICS])

    def write_data_to_file(self, run, model_metadata, data_outfile: Text = None, **kwargs):
        data_outfile = data_outfile or "run_" + \
                       datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        run_result = copy.deepcopy(run[RunKeys.RESULT_DATA])

        for i, res in enumerate(run_result):
            run_result[i] = convert_to_dict(res, model_metadata=model_metadata)

        run_data = pd.DataFrame(run_result)

        out_file = os.path.join(self.output_dir, data_outfile)

        run_data.to_csv(out_file, **kwargs)

    def visualize(self, run_id: Text, ax=None):

        df = self.return_result_data(run_id=run_id)

        df.plot(ax=ax)
