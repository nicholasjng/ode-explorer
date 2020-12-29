import logging
from typing import Dict, Text, Any

import numpy as np

from ode_explorer.models import BaseModel
from ode_explorer.types import ModelState

logger = logging.getLogger(__name__)


class Callback:
    """
    Base callback interface. Callbacks can be used to change the control flow of an
    integration run for an ODE model.
    """
    def __init__(self, name: Text = None):
        """
        Base callback constructor.

        Args:
            name: Optional string identifier.
        """
        self.__name__ = name or self.__class__.__name__

    def __call__(self,
                 i: int,
                 state: ModelState,
                 updated_state: ModelState,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> None:
        """
        Callback class call operator. Overload this with your custom logic to use in
        ODE integration runs.

        Args:
            i: Current iteration number.
            state: Previous ODE state.
            updated_state: New ODE state calculated by the used step function.
            model: ODE model that is used in the integration run.
            local_vars: Handle for the locals() dict passed to the Callbacks.
        """
        raise NotImplementedError


class NaNChecker(Callback):
    """
    Custom callback to check for NaNs in the ODE integration data.
    The occurrence of NaN values points to an instability in the
    step function or in the model function that was used.
    """
    def __init__(self,
                 nan_handling_mode: Text = "raise",
                 replacement: float = 0.0,
                 name: Text = None):
        """
        NaN checking callback constructor. The NaN handling mode is specified upon construction.
        Options are ``raise`` (a NaN value raises an error), ``coerce`` (NaN values are filled with a
        replacement value), or ``ignore`` (NaN values are propagated as-is).

        Args:
            nan_handling_mode: String, how to handle occurring NaN values.
            replacement: Replacement value for NaN values if nan_handling_mode was set to "coerce".
            name: Optional string identifier.
        """

        super(NaNChecker, self).__init__(name=name)

        self.replacement = replacement
        nan_handling_options = ["raise", "coerce", "ignore"]
        if nan_handling_mode not in nan_handling_options:
            raise ValueError("Unrecognized NaN handling mode supplied. "
                             f"Options are: {nan_handling_options}")
        self.nan_handling_mode = nan_handling_mode

    def __call__(self,
                 i: int,
                 state: ModelState,
                 updated_state: ModelState,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> None:
        """
        Control flow of the NaN checking callback.

        Args:
            i: Current iteration number.
            state: Previous ODE state.
            updated_state: New ODE state calculated by the used step function.
            model: ODE model that is used in the integration run.
            local_vars: Handle for the locals() dict passed to the Callbacks.

        Raises:
            ValueError: If any NaN values were found and the NaN handling mode was set to "raise".

        """

        # only check new state as old is assumed to be sensible,
        # i.e. not having nans
        y_new = updated_state[-1]

        na_mask = np.isnan(y_new)
        if np.any(na_mask):
            if self.nan_handling_mode == "raise":
                logger.error("Error: Encountered at least one NaN "
                             "value in the state after the ODE step.")

                raise ValueError("Error: There were NaN values in the ODE data"
                                 " and the NaN handling mode was set to "
                                 "\"{mode}\". If you want to fill NaN values"
                                 " with a fixed value instead, consider setting "
                                 "errors=\"coerce\" and supplying the \"replacement\" "
                                 "argument at callback construction.".format(mode=self.nan_handling_mode))

            elif self.nan_handling_mode == "coerce":
                logger.warning("Encountered at least one NaN "
                               "value in the state after the ODE "
                               "step. Filling the NaN values with "
                               "preset replacement value "
                               "{replacement}.".format(replacement=self.replacement))

                # get na_keys by na_mask
                y_new[na_mask] = self.replacement
                corrected = (*updated_state[:-1], y_new)
                local_vars.update({"updated_state": corrected})

            else:  # ignore errors
                logger.warning("Encountered at least one NaN "
                               "value in the state after the ODE "
                               "step. Since error handling mode "
                               "was set to {mode}, NaN values "
                               "will be ignored. This will most "
                               "likely have severe effects on your "
                               "computation.".format(mode=self.nan_handling_mode))
