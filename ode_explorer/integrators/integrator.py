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
from ode_explorer import constants
from ode_explorer.constants import RunKeys, RunConfigKeys
from ode_explorer.integrators.integrator_loops import constant_h_loop, dynamic_h_loop
from ode_explorer.metrics.metric import Metric
from ode_explorer.models.model import ODEModel
from ode_explorer.stepfunctions.templates import StepFunction
from ode_explorer.stepsize_control.stepsizecontroller import StepSizeController
from ode_explorer.types import ModelState
from ode_explorer.utils.data_utils import convert_to_dict
from ode_explorer.utils.run_utils import write_run_to_disk, get_run_metadata

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
                 base_output_dir: Text = None):

        # pre-step function, will be called before each step if specified
        self._pre_step_hook = pre_step_hook

        # empty list holding the different executed ODE integration runs
        self.runs = []

        # step count, can be used to track integration runs
        self._step_count = 0

        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self._set_up_logger(log_dir=self.log_dir)

        self.base_output_dir = base_output_dir or os.path.join(os.getcwd(), "results")

        if not os.path.exists(self.base_output_dir):
            os.mkdir(self.base_output_dir)

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

    def _flush_stale_file_handlers(self):
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename != self.logfile_name:
                    logger.handlers.remove(handler)

    @staticmethod
    def _make_run(model: ODEModel,
                  step_func: StepFunction,
                  sc: Union[StepSizeController, Callable],
                  initial_state: ModelState,
                  end: float,
                  h: float,
                  num_steps: int,
                  callbacks: List[Callback],
                  metrics: List[Metric]) -> Dict[Text, Any]:

        run = {constants.TIMESTAMP: datetime.datetime.now(),
               constants.RUN_ID: uuid.uuid4()}

        # initialize dimension names
        model.initialize_dim_names(initial_state)

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
                    RunKeys.RUN_CONFIG: run_config,
                    RunKeys.RESULT_DATA: [initial_state],
                    RunKeys.METRICS: [initial_metrics]})

        return run

    def integrate_const(self,
                        model: ODEModel,
                        step_func: StepFunction,
                        initial_state: ModelState,
                        end: float = None,
                        h: float = None,
                        num_steps: int = None,
                        reset: bool = False,
                        verbosity: int = 0,
                        output_dir: Text = None,
                        logfile: Text = None,
                        progress_bar: bool = False,
                        callbacks: List[Callback] = None,
                        metrics: List[Metric] = None):

        # callbacks and metrics
        callbacks = callbacks or []
        metrics = metrics or []

        if reset:
            self._reset()

        # create file handler
        if logfile:
            self._flush_stale_file_handlers()
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            logger.addHandler(fh)

        for handler in logger.handlers:
            handler.setLevel(verbosity)

        # construct run object
        run = self._make_run(model=model,
                             step_func=step_func,
                             initial_state=initial_state,
                             sc=None,
                             end=end,
                             h=h,
                             num_steps=num_steps,
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
                        metrics=metrics)

        logger.info("Finished integration.")

        if output_dir:
            self.save_run(run=run, output_dir=output_dir)

            logger.info("Run results saved to directory {}.".format(
                os.path.join(self.base_output_dir, output_dir)))

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
                              output_dir: Text = None,
                              logfile: Text = None,
                              progress_bar: bool = False,
                              callbacks: List[Callback] = None,
                              metrics: List[Metric] = None):

        # callbacks and metrics
        callbacks = callbacks or []
        metrics = metrics or []

        if reset:
            self._reset()

        # create file handler
        if logfile:
            self._flush_stale_file_handlers()
            fh = logging.FileHandler(os.path.join(self.log_dir, logfile))
            logger.addHandler(fh)

        for handler in logger.handlers:
            handler.setLevel(verbosity)

        # construct run object
        run = self._make_run(model=model,
                             step_func=step_func,
                             initial_state=initial_state,
                             sc=sc,
                             end=end,
                             h=initial_h,
                             num_steps=max_steps,
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
                       sc=sc)

        logger.info("Finished integration.")

        if output_dir:
            self.save_run(run=run, output_dir=output_dir)

            logger.info("Run results saved to directory {}.".format(
                os.path.join(self.base_output_dir, output_dir)))

        self.runs.append(run)

        return self

    def list_runs(self, tablefmt: Text = "github"):
        if len(self.runs) == 0:
            print("No runs available!")
            return

        metadata_list = [get_run_metadata(run) for run in self.runs]

        print(tabulate(metadata_list, headers="keys", tablefmt=tablefmt))

    def get_run_by_id(self, run_id: Text):
        if len(self.runs) == 0:
            raise ValueError("No runs available. Please integrate a model first!")
        if run_id == "latest":
            return self.runs[-1]
        try:
            run = next(r for r in self.runs if run_id in str(r[constants.RUN_ID]))
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

    def save_run(self, run: Dict, output_dir):
        out_dir = os.path.join(self.base_output_dir, output_dir)
        write_run_to_disk(run=run, out_dir=out_dir)

    def visualize(self, run_id: Text, ax=None):

        df = self.return_result_data(run_id=run_id)

        df.plot(ax=ax)
