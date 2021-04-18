import datetime
import logging
import os
import uuid
from typing import Dict, Callable, Text, List, Union, Any

import absl.logging
from tabulate import tabulate

from ode_explorer.callbacks import Callback
from ode_explorer.constants import ResultKeys, ConfigKeys
from ode_explorer.integrators.loop_factory import loop_factory
from ode_explorer.metrics import Metric
from ode_explorer.models import BaseModel
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.stepsize_control import StepSizeController
from ode_explorer.types import State
from ode_explorer.utils.result_utils import get_result_metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(absl.logging.PythonFormatter())
logger.addHandler(ch)


class Integrator:
    """
    Base class for all ODE integrators. An integrator keeps minimal state to facilitate IO and
    logging of model integration results. It also serves as a registry for all model results and can be
    queried for specific results by different attributes.
    """

    def __init__(self,
                 base_log_dir: Text = None,
                 logfile_name: Text = None,
                 base_output_dir: Text = None):
        """
        Base Integrator constructor.

        Args:
            base_log_dir: Base directory for saving ODE integration logs.
            logfile_name: Base log file object to save all logs into.
            base_output_dir: Base output directory for saving result and model data.
        """
        # empty list holding the different executed ODE integration results
        self.results = []

        self.base_log_dir = base_log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self._set_up_logger(log_dir=self.base_log_dir)

        self.base_output_dir = base_output_dir or os.path.join(os.getcwd(), "results")

        if not os.path.exists(self.base_output_dir):
            os.mkdir(self.base_output_dir)

        logger.info("Created an Integrator instance.")

    def _reset(self):
        # Hard reset all data
        self.results = []

    def _set_up_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        fh = logging.FileHandler(os.path.join(self.base_log_dir, self.logfile_name))
        fh.setLevel(logging.INFO)
        fh.setFormatter(absl.logging.PythonFormatter())
        logger.addHandler(fh)
        logger.info('Creating an Integrator instance.')

    def _flush_stale_file_handlers(self):
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename != self.logfile_name:
                logger.handlers.remove(handler)

    def _make_config(self,
                     model: BaseModel,
                     step_func: StepFunction,
                     sc: Union[StepSizeController, Callable],
                     initial_state: State,
                     end: float,
                     h: float,
                     max_steps: int,
                     callbacks: List[Callback],
                     metrics: List[Metric]) -> Dict[Text, Any]:

        start, _ = initial_state

        config = {ConfigKeys.TIMESTAMP: datetime.datetime.now().strftime("%c"),
                  ConfigKeys.ID: str(uuid.uuid4()),
                  ConfigKeys.LOOP_TYPE: "static" if not bool(sc) else "dynamic",
                  ConfigKeys.START: start,
                  ConfigKeys.END: end,
                  ConfigKeys.STEP_SIZE: h,
                  ConfigKeys.NUM_STEPS: max_steps,
                  ConfigKeys.METRICS: [m.__name__ for m in metrics],
                  ConfigKeys.CALLBACKS: [c.__name__ for c in callbacks]
                  }

        return config

    def _integrate(self,
                   loop_type: Text,
                   model: BaseModel,
                   step_func: StepFunction,
                   initial_state: State,
                   end: float,
                   verbosity: int = 0,
                   output_dir: Text = None,
                   logfile: Text = None,
                   progress_bar: bool = False,
                   **loop_kwargs):

        # create file handler
        if logfile:
            self._flush_stale_file_handlers()
            fh = logging.FileHandler(os.path.join(self.base_log_dir, logfile))
            logger.addHandler(fh)

        for handler in logger.handlers:
            handler.setLevel(verbosity)

        # construct result object
        config = self._make_config(model=model,
                                   step_func=step_func,
                                   initial_state=initial_state,
                                   end=end,
                                   **loop_kwargs)

        logger.info("Starting integration.")

        result = loop_factory.get(loop_type)(step_func=step_func,
                                             model=model,
                                             initial_state=initial_state,
                                             progress_bar=progress_bar,
                                             **loop_kwargs)

        logger.info("Finished integration.")

        result_dict = {ResultKeys.RESULT_DATA: result,
                       ResultKeys.CONFIG: config}

        self.results.append(result_dict)

        if output_dir:
            self.save_result(result=result, output_dir=output_dir)

            logger.info("Results saved to directory {}.".format(
                os.path.join(self.base_output_dir, output_dir)))

        return self

    def integrate_const(self,
                        model: BaseModel,
                        step_func: StepFunction,
                        initial_state: State,
                        end: float = None,
                        h: float = None,
                        max_steps: int = None,
                        reset: bool = False,
                        verbosity: int = logging.INFO,
                        output_dir: Text = None,
                        logfile: Text = None,
                        progress_bar: bool = False,
                        callbacks: List[Callback] = None,
                        metrics: List[Metric] = None):
        """
        Integrate a model with a chosen step function and a constant step size.

        Args:
            model: ODEModel instance of your ODE problem.
            step_func: Step Function used to integrate the model.
            initial_state: State tuple containing the initial state variables.
            end: Target end time for ODE solving. Equals the time value of the last step.
            h: Constant step size for integration.
            max_steps: Maximum allowed steps during the integration.
            reset: Bool, whether to reset the integrator (this deletes all previous results).
            verbosity: Logging verbosity, default logging.INFO.
            output_dir: Output directory. If specified,saves result data and info into this directory.
            logfile: Log file. If specified, writes all logs of the integration into this file.
            progress_bar: Bool, whether to display a progress bar during the result.
            callbacks: List of callbacks to execute after each step.
            metrics: List of metrics to calculate after each step.
        """
        # empty lists in case nothing was supplied
        callbacks = callbacks or []
        metrics = metrics or []

        return self._integrate(loop_type="constant",
                               model=model,
                               step_func=step_func,
                               initial_state=initial_state,
                               end=end,
                               h=h,
                               max_steps=max_steps,
                               reset=reset,
                               verbosity=verbosity,
                               output_dir=output_dir,
                               logfile=logfile,
                               progress_bar=progress_bar,
                               callbacks=callbacks,
                               metrics=metrics,
                               sc=None)

    def integrate_adaptively(self,
                             model: BaseModel,
                             step_func: StepFunction,
                             initial_state: State,
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
        """
        Integrate a model with a chosen step function adaptively with custom step size control.

        Args:
            model: ODEModel instance of your ODE problem.
            step_func: Step Function used to integrate the model.
            initial_state: State tuple containing the initial state variables.
            sc: Step size controller, adjusting the step size throughout the integration.
            end: Target end time for ODE solving. Equals the time value of the last step.
            initial_h: Initial step size for integration.
            max_steps: Maximum allowed steps during the integration.
            reset: Bool, whether to reset the integrator (this deletes all previous results).
            verbosity: Logging verbosity, default logging.INFO.
            output_dir: Output directory. If specified,saves result data and info into this directory.
            logfile: Log file. If specified, writes all logs of the integration into this file.
            progress_bar: Bool, whether to display a progress bar during the result.
            callbacks: List of callbacks to execute after each step.
            metrics: List of metrics to calculate after each step.
        """
        # empty lists in case nothing was supplied
        callbacks = callbacks or []
        metrics = metrics or []

        return self._integrate(loop_type="adaptive",
                               model=model,
                               step_func=step_func,
                               initial_state=initial_state,
                               end=end,
                               h=initial_h,
                               max_steps=max_steps,
                               reset=reset,
                               verbosity=verbosity,
                               output_dir=output_dir,
                               logfile=logfile,
                               progress_bar=progress_bar,
                               callbacks=callbacks,
                               metrics=metrics,
                               sc=sc)

    def list_results(self, tablefmt: Text = "github"):
        """
        Lists metadata of all available previous results.

        Args:
            tablefmt: Table format, passed to tabulate.
        """

        if len(self.results) == 0:
            print("No results available!")
            return

        metadata_list = [get_result_metadata(result) for result in self.results]

        print(tabulate(metadata_list, headers="keys", tablefmt=tablefmt))

    def get_result_by_id(self, result_id: Text):
        """
        Returns a previous ODE integration result by (partial) ID.

        Args:
            result_id: ID of the chosen integration result object.

        Raises:
            ValueError: If no result matches the given result ID.

        """
        if len(self.results) == 0:
            raise ValueError("No results available. Please integrate a model first!")
        if result_id == "latest":
            return self.results[-1][ResultKeys.RESULT_DATA]
        try:
            result = next(r for r in self.results if result_id in str(r[ResultKeys.CONFIG][ConfigKeys.ID]))
        except StopIteration:
            raise ValueError(f"Result with ID {result_id} not found.")

        return result[ResultKeys.RESULT_DATA]

    def return_result_data(self, result_id: Text) -> List:
        """
        Return data of a previous integration result.

        Args:
            result_id: ID of the chosen integration result object.

        Returns:
            A list containing the ODE integration data for each step.
        """
        return self.get_result_by_id(result_id=result_id)

    def return_metrics(self, result_id: Text) -> List:
        """
         Return metrics data of a previous integration result.

        Args:
            result_id: ID of the chosen integration result object.

        Returns:
            A list of the metric data for each step of the result with ID result_id.
        """
        raise NotImplementedError

    def save_result(self, result: List, output_dir):
        """
        Saves a result object to an output directory on disk.

        Args:
            result: result object obtained as output from ODE integration.
            output_dir: Target directory to save the result to.

        """
        raise NotImplementedError
        # out_dir = os.path.join(self.base_output_dir, output_dir)
        # write_result_to_disk(result=result, out_dir=out_dir)
