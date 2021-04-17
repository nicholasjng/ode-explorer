import copy
import datetime
import logging
import os
import uuid
from typing import Dict, Callable, Text, List, Union, Any

import absl.logging
import pandas as pd
from tabulate import tabulate

from ode_explorer import constants
from ode_explorer.callbacks import Callback
from ode_explorer.constants import ResultKeys, ConfigKeys, ModelMetadataKeys
from ode_explorer.integrators.loop_factory import loop_factory
from ode_explorer.metrics import Metric
from ode_explorer.models import BaseModel
from ode_explorer.stepfunctions import StepFunction
from ode_explorer.stepsize_control import StepSizeController
from ode_explorer.types import ModelState
from ode_explorer.utils.data_utils import convert_to_dict, initialize_dim_names
from ode_explorer.utils.result_utils import write_result_to_disk, get_result_metadata

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
                 base_output_dir: Text = None,
                 csv_io_args: Dict[Text, Any] = None):
        """
        Base Integrator constructor.

        Args:
            base_log_dir: Base directory for saving ODE integration logs.
            logfile_name: Base log file object to save all logs into.
            base_output_dir: Base output directory for saving result and model data.
            csv_io_args: Additional keyword arguments passed to pandas.DataFrame.to_csv
             when writing data to a CSV file.
        """
        # empty list holding the different executed ODE integration results
        self.results = []

        self.base_log_dir = base_log_dir or os.path.join(os.getcwd(), "logs")

        self.logfile_name = logfile_name or "logs.txt"

        self._set_up_logger(log_dir=self.base_log_dir)

        self.base_output_dir = base_output_dir or os.path.join(os.getcwd(), "results")

        self.datetime_format = "%c"

        self.csv_io_args = csv_io_args or {}

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
                     # step_func: StepFunction,
                     # sc: Union[StepSizeController, Callable],
                     initial_state: ModelState,
                     end: float,
                     h: float,
                     max_steps: int,
                     callbacks: List[Callback],
                     metrics: List[Metric]) -> Dict[Text, Any]:

        # callbacks and metrics
        callbacks = callbacks or []
        metrics = metrics or []

        result = {constants.TIMESTAMP: datetime.datetime.now().strftime(self.datetime_format),
                  constants.RESULT_ID: str(uuid.uuid4())}

        start = initial_state[0]

        result_config = {ConfigKeys.START: start,
                         ConfigKeys.END: end,
                         ConfigKeys.STEP_SIZE: h,
                         ConfigKeys.NUM_STEPS: max_steps,
                         ConfigKeys.METRICS: [m.__name__ for m in metrics],
                         ConfigKeys.CALLBACKS: [c.__name__ for c in callbacks]
                         }

        # initial_metrics = {}
        #
        # for metric in metrics:
        #     val = metric(0, initial_state, initial_state, model, locals())
        #     initial_metrics[metric.__name__] = val

        # if bool(sc):
        #     # means dynamical integration, hence we log step size, accepts and rejects
        #     initial_metrics.update({defaults.iteration: 0,
        #                             defaults.step_size: h,
        #                             defaults.accepted: 1,
        #                             defaults.rejected: 0})

        result.update({ResultKeys.MODEL_METADATA: model.get_metadata(),
                       ResultKeys.CONFIG: result_config})
        return result

    def _integrate(self,
                   loop_type: Text,
                   model: BaseModel,
                   step_func: StepFunction,
                   initial_state: ModelState,
                   end: float,
                   reset: bool = False,
                   verbosity: int = 0,
                   output_dir: Text = None,
                   logfile: Text = None,
                   progress_bar: bool = False,
                   **loop_kwargs):

        if reset:
            self._reset()

        # create file handler
        if logfile:
            self._flush_stale_file_handlers()
            fh = logging.FileHandler(os.path.join(self.base_log_dir, logfile))
            logger.addHandler(fh)

        for handler in logger.handlers:
            handler.setLevel(verbosity)

        # construct result object
        result = self._make_config(model=model,
                                   # step_func=step_func,
                                   initial_state=initial_state,
                                   end=end,
                                   **loop_kwargs)

        # deepcopy here, otherwise the initial state gets overwritten
        state = copy.deepcopy(initial_state)

        logger.info("Starting integration.")

        result = loop_factory.get(loop_type)(step_func=step_func,
                                             model=model,
                                             state=state,
                                             progress_bar=progress_bar,
                                             **loop_kwargs)

        logger.info("Finished integration.")

        if output_dir:
            self.save_result(result=result, output_dir=output_dir)

            logger.info("Results saved to directory {}.".format(
                os.path.join(self.base_output_dir, output_dir)))

        self.results.append(result)

        return self

    def integrate_const(self,
                        model: BaseModel,
                        step_func: StepFunction,
                        initial_state: ModelState,
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
            return self.results[-1]
        try:
            result = next(r for r in self.results if result_id in str(r[constants.RESULT_ID]))
        except StopIteration:
            raise ValueError(f"Result with ID {result_id} not found.")

        return result

    def return_result_data(self, result_id: Text) -> pd.DataFrame:
        """
        Construct a pd.DataFrame out of the result data of a previous integration result.

        Args:
            result_id: ID of the chosen integration result object.

        Returns:
            A pd.DataFrame containing the integration data as rows.
        """

        result = self.get_result_by_id(result_id=result_id)

        model_metadata = result[ResultKeys.MODEL_METADATA]

        dim_names = model_metadata[ModelMetadataKeys.DIM_NAMES]

        variable_names = model_metadata[ModelMetadataKeys.VARIABLE_NAMES]

        result_copy = copy.deepcopy(result[ResultKeys.RESULT_DATA])

        if not dim_names:
            dim_names = initialize_dim_names(variable_names, result_copy[0])

        for i, res in enumerate(result_copy):
            result_copy[i] = convert_to_dict(res,
                                             model_metadata=model_metadata,
                                             dim_names=dim_names)

        return pd.DataFrame(result_copy)

    def return_metrics(self, result_id: Text) -> pd.DataFrame:
        """
        Construct a pd.DataFrame out of the result data of a previous integration result.

        Args:
            result_id: ID of the chosen integration result object.

        Returns:
            A pd.DataFrame containing the metric data of the result with ID result_id as rows.
        """

        result = self.get_result_by_id(result_id=result_id)

        return pd.DataFrame(result[ResultKeys.METRICS])

    def save_result(self, result: List, output_dir):
        """
        Saves a result object to an output directory on disk.

        Args:
            result: result object obtained as output from ODE integration.
            output_dir: Target directory to save the result to.

        """
        out_dir = os.path.join(self.base_output_dir, output_dir)
        write_result_to_disk(result=result, out_dir=out_dir, **self.csv_io_args)
