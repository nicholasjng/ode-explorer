import numpy as np

from ode_explorer.model import ODEModel
from ode_explorer.constants import VARIABLES, ZIPPED
from typing import Text, Dict, Union
from utils.data_utils import is_scalar


class StepFunction:
    """
    Base class for all ODE step functions.
    """

    def __init__(self, output_format: Text = VARIABLES, order: int = 0):
        # order of the method
        self.order = order
        if output_format not in [VARIABLES, ZIPPED]:
            raise ValueError(f"Error: Output format \"{output_format}\" not "
                             f"understood.")
        self.output_format = output_format

    @staticmethod
    def get_data_from_state_dict(model: ODEModel,
                                 state_dict: Dict[Text,
                                                  Union[np.ndarray, float]],
                                 input_format: Text):

        t = state_dict.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        if input_format == VARIABLES:
            y = state_dict[model.variable_names[0]]
        elif input_format == ZIPPED:
            y = np.array(list(state_dict.values()))
        else:
            raise ValueError(f"Error: Input format {input_format} not "
                             f"understood.")

        # scalar ODE, return just the value then
        if not is_scalar(y) and len(y) == 1:
            return t, y[0]

        return t, y

    @staticmethod
    def make_zipped_dict(model: ODEModel, t: float,
                         y: Union[np.ndarray, float]):
        if is_scalar(y):
            y_new = [y]
        else:
            y_new = y

        return {**{model.indep_name: t},
                **dict(zip(model.dim_names, y_new))}

    @staticmethod
    def make_state_dict(model: ODEModel, t: float,
                        y: Union[np.ndarray, float]):
        return {model.indep_name: t, model.variable_names[0]: y}

    def make_new_state_dict(self, model: ODEModel, t: float,
                            y: Union[np.ndarray, float]):

        if self.output_format == ZIPPED:
            return self.make_zipped_dict(model=model, t=t, y=y)
        else:
            return self.make_state_dict(model=model, t=t, y=y)

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        raise NotImplementedError

class ExplicitRungeKuttaMethod(StepFunction):
    def __init(self,
               alphas: np.ndarray,
               betas: np.ndarray,
               gammas: np.ndarray,
               output_format: Text = VARIABLES,
               order: int = 0):

        super(ExplicitRungeKuttaMethod, self).__init__(output_format,
                                                       order)

        self.validate_butcher_tableau(alphas=alphas, betas=betas,
                                      gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.order = order
        self.num_stages = len(self.alphas)
        self.ks = np.zeros((1, betas.shape[0]))

    @staticmethod
    def validate_butcher_tableau(alphas: np.ndarray,
                                 betas: np.ndarray,
                                 gammas: np.ndarray) -> None:
        _error_msg = []
        if len(alphas) != len(gammas):
            _error_msg.append("Alpha and gamma vectors are "
                              "not the same length")

        if betas.shape[0] != betas.shape[1]:
            _error_msg.append("Betas must be a quadratic matrix with the same "
                              "dimension as the alphas/gammas array")

        if _error_msg:
            raise ValueError("Error while validating the input Butcher "
                             "tableau. More information: "
                             "{}".format(",".join(_error_msg)))

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict,
                                             input_format=input_format)

        if not is_scalar(y) and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), 4))

        ha = self.alphas * h
        hg = self.gammas * h
        ks = self.ks

        # ks[:, 0] = model(t, y, **kwargs)

        for i in range(self.num_stages):
            # first row of betas is a zero row
            # because it is an explicit RK
            ks[:, i] = model(t + ha[i], y + ha[i] * self.betas[i] * ks)

        y_new = y + np.sum(ks * hg, axis=1)

        new_state = self.make_new_state_dict(model=model, t=t+h, y=y_new)

        return new_state
