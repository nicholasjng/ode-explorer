import logging
from typing import Dict, Text, Any

import numpy as np

from ode_explorer.models.model import ODEModel
from ode_explorer.types import ModelState

logger = logging.getLogger(__name__)


class Callback:
    def __init__(self, name: Text = None):
        self.__name__ = name or self.__class__.__name__

    def __call__(self,
                 i: int,
                 state: ModelState,
                 updated_state: ModelState,
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> None:
        raise NotImplementedError


class NaNChecker(Callback):
    def __init__(self,
                 fill_nans: bool = False,
                 replacement: float = 0.0,
                 nan_handling_mode: Text = "raise",
                 name: Text = None):

        super(NaNChecker, self).__init__(name=name)

        self.fill_nans = fill_nans
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
                 model: ODEModel,
                 local_vars: Dict[Text, Any]) -> None:

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
                print(corrected)

            else:  # ignore errors
                logger.warning("Encountered at least one NaN "
                               "value in the state after the ODE "
                               "step. Since error handling mode "
                               "was set to {mode}, NaN values "
                               "will be ignored. This will most "
                               "likely have severe effects on your "
                               "computation.".format(mode=self.nan_handling_mode))
