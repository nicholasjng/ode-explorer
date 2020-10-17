import numpy as np
from ode_explorer.integrator import Integrator
from ode_explorer.model import ODEModel

from typing import Dict, Text, Any


class Callback:
    def __call__(self,
                 i: int,
                 integrator: Integrator,
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> None:
        raise NotImplementedError


class NaNChecker(Callback):
    def __init(self, fill_nans: bool = False, replacement: float = 0.0,
               errors: Text = "raise"):
        self.fill_nans = fill_nans
        self.replacement = replacement
        error_options = ["raise", "coerce", "ignore"]
        if errors not in error_options:
            raise ValueError("Unrecognized error handling mode supplied. "
                             f"Options are: {error_options}")
        self.errors = errors

    def __call__(self,
                 i: int,
                 integrator: Integrator,
                 model: ODEModel,
                 locals: Dict[Text, Any]) -> None:

        state_dict = locals["state_dict"]
        updated_state_dict = locals["updated_state_dict"]

        # only check new state as old is assumed to be sensible,
        # i.e. not having nans
        keys = list(updated_state_dict.keys())
        y_new = np.array(updated_state_dict.values())

        na_mask = np.isnan(y_new)
        if np.any(na_mask):
            if self.errors == "raise":
                integrator.logger.error("Error: Encountered at least one NaN "
                                        "value in the state after the ODE "
                                        "step.")
                raise ValueError("Error: There were NaN values in the ODE data"
                                 " and the error handling mode was set to "
                                 "\"{errors}\". If you want to fill NaN values"
                                 " with a fixed value instead, consider "
                                 "supplying the \"replacement\" argument at "
                                 "callback construction.".format(
                                  errors=self.errors))
            elif self.errors == "coerce":
                integrator.logger.warning("Encountered at least one NaN "
                                          "value in the state after the ODE "
                                          "step. Filling the NaN values with "
                                          "preset replacement value "
                                          "{replacement}.".format(
                                           replacement=self.replacement))

                # get na_keys by na_mask
                na_keys = [key for i, key in enumerate(keys) if na_mask[i]]
                for na_key in na_keys:
                    updated_state_dict[na_key] = self.replacement

            else:  # ignore errors
                integrator.logger.warning("Encountered at least one NaN "
                                          "value in the state after the ODE "
                                          "step. Since error handling mode "
                                          "was set to {errors}, NaN values "
                                          "will be ignored. This will most "
                                          "likely have severe effects on your "
                                          "computation.".format(
                                           errors=self.errors))
