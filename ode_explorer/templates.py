import numpy as np
import logging

from ode_explorer.model import ODEModel
from ode_explorer.constants import DataFormatKeys
from typing import Text, Dict, Union
from ode_explorer.utils.helpers import is_scalar
from scipy.optimize import root, root_scalar

logging.basicConfig(level=logging.DEBUG)
templates_logger = logging.getLogger("ode_explorer.templates")


class StepFunction:
    """
    Base class for all ODE step functions.
    """

    def __init__(self,
                 output_format: Text = DataFormatKeys.VARIABLES,
                 order: int = 0):
        # order of the method
        self.order = order
        if output_format not in [DataFormatKeys.VARIABLES, DataFormatKeys.ZIPPED]:
            raise ValueError(f"Error: Output format \"{output_format}\" not "
                             f"understood.")
        self.output_format = output_format

    @staticmethod
    def get_data_from_state(model: ODEModel,
                            state: Dict[Text, Union[np.ndarray, float]],
                            input_format: Text):

        t = state.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        if input_format == DataFormatKeys.VARIABLES:
            y = state[model.variable_names[0]]
        elif input_format == DataFormatKeys.ZIPPED:
            y = np.array(list(state.values()))
        else:
            raise ValueError(f"Error: Input format {input_format} not "
                             f"understood.")

        # scalar ODE, return just the value then
        if not is_scalar(y) and len(y) == 1:
            return t, y[0]

        return t, y

    @staticmethod
    def make_zipped_dict(model: ODEModel, t: float,
                         y: Union[np.ndarray, float]) -> \
            Dict[Text, Union[np.ndarray, float]]:
        if is_scalar(y):
            y_new = [y]
        else:
            y_new = y

        return {**{model.indep_name: t},
                **dict(zip(model.dim_names, y_new))}

    @staticmethod
    def make_state_dict(model: ODEModel, t: float,
                        y: Union[np.ndarray, float]) -> \
            Dict[Text, Union[np.ndarray, float]]:
        return {model.indep_name: t, model.variable_names[0]: y}

    def make_new_state(self, model: ODEModel, t: float,
                       y: Union[np.ndarray, float]) -> \
            Dict[Text, Union[np.ndarray, float]]:

        if self.output_format == DataFormatKeys.ZIPPED:
            return self.make_zipped_dict(model=model, t=t, y=y)
        else:
            return self.make_state_dict(model=model, t=t, y=y)

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = DataFormatKeys.VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        raise NotImplementedError


class ExplicitRungeKuttaMethod(StepFunction):
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 output_format: Text = DataFormatKeys.VARIABLES,
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
                              "dimension as the alphas/gammas arrays")

        # for an explicit method, betas must be lower triangular
        if not np.allclose(betas, np.tril(betas, k=-1)):
            _error_msg.append("The beta matrix has to be lower triangular for "
                              "an explicit Runge-Kutta method, i.e. "
                              "b_ij = 0 for i <= j")

        if _error_msg:
            raise ValueError("An error occurred while validating the input "
                             "Butcher tableau. More information: "
                             "{}.".format(",".join(_error_msg)))

    def forward(self,
                model: ODEModel,
                state: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = DataFormatKeys.VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        if not is_scalar(y) and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), self.num_stages))

        ha = self.alphas * h
        hg = self.gammas * h
        ks = self.ks

        # ks[:, 0] = model(t, y, **kwargs)

        for i in range(self.num_stages):
            # first row of betas is a zero row
            # because it is an explicit RK
            ks[:, i] = model(t + ha[i], y + ha[i] * ks.dot(self.betas[i]))

        y_new = y + np.sum(ks * hg, axis=1)

        new_state = self.make_new_state(model=model, t=t + h, y=y_new)

        return new_state


class ImplicitRungeKuttaMethod(StepFunction):
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 output_format: Text = DataFormatKeys.VARIABLES,
                 order: int = 0,
                 **kwargs):

        super(ImplicitRungeKuttaMethod, self).__init__(output_format,
                                                       order)

        self.validate_butcher_tableau(alphas=alphas, betas=betas,
                                      gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.order = order
        self.num_stages = len(self.alphas)
        self.ks = np.zeros((1, betas.shape[0]))

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

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
                              "dimension as the alphas/gammas arrays")

        if _error_msg:
            raise ValueError("An error occurred while validating the input "
                             "Butcher tableau. More information: "
                             "{}.".format(",".join(_error_msg)))

    def forward(self,
                model: ODEModel,
                state: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = DataFormatKeys.VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        if not is_scalar(y) and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), self.num_stages))

        ha = self.alphas * h
        hb = self.betas * h
        hg = self.gammas * h
        ks = self.ks

        n, m = ks.shape

        def F(x: np.ndarray, *args) -> np.ndarray:

            # kwargs are not allowed in scipy.optimize, so pass tuple instead
            # TODO: Vectorizing this needs to happen in JAX, that would be
            #  amazing. Also adding a Jacobian
            model_stack = np.hstack(model(t + ha[i],
                                          x.reshape((n, m)).dot(hb[i]),
                                          *args) for i in range(m))

            return model_stack - x

        # modified function in case of using Implicit Euler method or
        # equivalents on a scalar ODE
        def F_scalar(x: float, *args) -> float:
            return model(t + ha[0], y + hb[0] * x, *args) - x

        # this bit is important to sort the kwargs before putting them into
        # the tuple passed to root
        # model_spec = inspect.getfullargspec(model.ode_fn).args[2:]
        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        if n * m != 1:
            # TODO: Retry here in case of convergence failure?
            root_res = root(F, x0=ks.reshape((n*m,)),
                            args=args, **self.solver_kwargs)
            # this line ensures that np.sum returns a scalar for a scalar ODE
            axis = None if is_scalar(y) else 1

            y_new = y + np.sum(root_res.x.reshape((n, m)) * hg, axis=axis)

        else:
            root_res = root_scalar(F_scalar, args=args, **self.solver_kwargs)

            y_new = y + hg[0] * root_res.x

        new_state = self.make_new_state(model=model, t=t + h, y=y_new)

        return new_state


class ExplicitMultistepMethod(StepFunction):
    """
        Adams-Bashforth Method of order 2 for ODE solving.
        """

    def __init__(self,
                 startup: StepFunction,
                 b_coeffs: np.ndarray,
                 output_format: Text = DataFormatKeys.VARIABLES,
                 order: int = 0):
        super(ExplicitMultistepMethod, self).__init__(output_format=output_format)
        self.order = order
        # startup calculation variables, only for multistep methods
        self.ready = False
        self.startup = startup

        # multistep method variables
        self.a_coeffs = np.array([1.0])  # unused
        self.b_coeffs = b_coeffs

        self.lookback = len(b_coeffs)
        # side cache for additional steps
        self.y_cache = np.ones((1, self.lookback))
        self.t_cache = np.zeros(self.lookback)
        self.f_cache = np.zeros_like(self.y_cache)

    def reset(self):
        # Resets the run so that next time the step function is called,
        # new startup values will be calculated with the saved startup step
        # function. Useful if the step function is supposed to be reused in
        # multiple non-consecutive runs.
        self.ready = False

    def perform_startup_calculation(self, model: ODEModel,
                                    state: Dict[Text, Union[np.ndarray, float]],
                                    h: float,
                                    input_format: Text = DataFormatKeys.VARIABLES,
                                    **kwargs):

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        if not is_scalar(y) and len(y) != self.y_cache.shape[0]:
            self.y_cache = np.zeros((len(y), self.lookback))

        self.t_cache[0], self.y_cache[:, 0] = t, y
        # fill function evaluation cache
        self.f_cache[:, 0] = model(t, y, **kwargs)

        dummy_dict = state.copy()
        for i in range(1, self.lookback):
            startup_state = self.startup.forward(model=model,
                                                 state_dict=dummy_dict,
                                                 h=h, **kwargs)

            t1, y1 = self.get_data_from_state(model=model,
                                              state=startup_state,
                                              input_format=input_format)

            self.t_cache[i], self.y_cache[:, i] = t1, y1
            self.f_cache[:, i] = model(t1, y1, **kwargs)
            dummy_dict.update(startup_state)

        self.ready = True

    def get_cached_values(self, t: float):
        eps = 1e-15
        closest_in_cache = np.isclose(self.t_cache, t)
        idx = np.argmax(closest_in_cache) + 1
        if not any(closest_in_cache):
            idx = np.argmax(self.t_cache > t + eps)

        return self.t_cache[idx], self.y_cache[idx]

    def forward(self,
                model: ODEModel,
                state: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = DataFormatKeys.VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        if not self.ready:
            # startup calculation to the multistep method,
            # fills the y-, t- and f-caches
            self.perform_startup_calculation(model=model,
                                             state=state,
                                             h=h,
                                             **kwargs)

            # first cached value
            y_new = self.y_cache[:, 1]

            new_state = self.make_new_state(model=model,
                                            t=self.t_cache[1],
                                            y=y_new)

            return new_state

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        # This branch is curious
        eps = 1e-12
        if t + eps < self.t_cache[-1]:
            t_new, y_new = self.get_cached_values(t)

            new_state = self.make_new_state(model=model, t=t_new, y=y_new)

            return new_state

        y_new = y + h * np.sum(self.b_coeffs * self.f_cache, axis=1)

        # shift all y and all f evaluations to the left by 1,
        # we only need the two previous steps
        self.y_cache = np.roll(self.y_cache, shift=-1, axis=1)
        self.f_cache = np.roll(self.f_cache, shift=-1, axis=1)

        self.y_cache[:, -1] = y_new
        self.f_cache[:, -1] = model(t + h, y_new, **kwargs)

        new_state = self.make_new_state(model=model, t=t + h, y=y_new)

        return new_state


class ImplicitMultistepMethod(StepFunction):
    """
        Adams-Bashforth Method of order 2 for ODE solving.
        """

    def __init__(self,
                 startup: StepFunction,
                 b_coeffs: np.ndarray,
                 output_format: Text = DataFormatKeys.VARIABLES,
                 order: int = 0,
                 **kwargs):
        super(ImplicitMultistepMethod, self).__init__(output_format=output_format)
        self.order = order
        # startup calculation variables, only for multistep methods
        self.ready = False
        self.startup = startup

        # multistep method variables
        self.a_coeffs = np.array([1.0])  # unused
        self.b_coeffs = b_coeffs

        self.lookback = len(b_coeffs)
        # side cache for additional steps
        self.y_cache = np.ones((1, self.lookback))
        self.t_cache = np.zeros(self.lookback)
        self.f_cache = np.zeros_like(self.y_cache)

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def reset(self):
        # Resets the run so that next time the step function is called,
        # new startup values will be calculated with the saved startup step
        # function. Useful if the step function is supposed to be reused in
        # multiple non-consecutive runs.
        self.ready = False

    def perform_startup_calculation(self,
                                    model: ODEModel,
                                    state: Dict[Text, Union[np.ndarray, float]],
                                    h: float,
                                    input_format: Text = DataFormatKeys.VARIABLES,
                                    **kwargs):

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        if not is_scalar(y) and len(y) != self.y_cache.shape[0]:
            self.y_cache = np.zeros((len(y), self.lookback))

        self.t_cache[0], self.y_cache[:, 0] = t, y
        # fill function evaluation cache
        self.f_cache[:, 0] = model(t, y, **kwargs)

        dummy_dict = state.copy()
        for i in range(1, self.lookback):
            startup_state = self.startup.forward(model=model,
                                                 state=dummy_dict,
                                                 h=h, **kwargs)

            t1, y1 = self.get_data_from_state(model=model,
                                              state=startup_state,
                                              input_format=input_format)

            self.t_cache[i], self.y_cache[:, i] = t1, y1
            self.f_cache[:, i] = model(t1, y1, **kwargs)
            dummy_dict.update(startup_state)

        self.ready = True

    def get_cached_values(self, t: float):
        eps = 1e-15
        closest_in_cache = np.isclose(self.t_cache, t)
        idx = np.argmax(closest_in_cache) + 1
        if not any(closest_in_cache):
            idx = np.argmax(self.t_cache > t + eps)

        return self.t_cache[idx], self.y_cache[idx]

    def forward(self,
                model: ODEModel,
                state: Dict[Text, Union[np.ndarray, float]],
                h: float,
                input_format: Text = DataFormatKeys.VARIABLES,
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        if not self.ready:
            # startup calculation to the multistep method,
            # fills the y-, t- and f-caches
            self.perform_startup_calculation(model=model,
                                             state=state,
                                             h=h,
                                             **kwargs)

            # first cached value
            y_new = self.y_cache[:, 1]

            new_state = self.make_new_state(model=model,
                                            t=self.t_cache[1],
                                            y=y_new)

            return new_state

        t, y = self.get_data_from_state(model=model,
                                        state=state,
                                        input_format=input_format)

        # This branch is curious
        eps = 1e-12
        if t + eps < self.t_cache[-1]:
            t_new, y_new = self.get_cached_values(t)

            new_state = self.make_new_state(model=model, t=t_new, y=y_new)

            return new_state

        def F(x: Union[np.ndarray, float], *args):
            return h * model(t+h, x, *args) + y - \
                   h * np.sum(self.b_coeffs * self.f_cache, axis=1) - x

        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        root_res = root(F, x0=self.f_cache[:, -1],
                        args=args, **self.solver_kwargs)

        y_new = root_res.x

        # shift all y and all f evaluations to the left by 1,
        # we only need the two previous steps
        self.y_cache = np.roll(self.y_cache, shift=-1, axis=1)
        self.f_cache = np.roll(self.f_cache, shift=-1, axis=1)

        self.y_cache[:, -1] = y_new
        self.f_cache[:, -1] = model(t + h, y_new, **kwargs)

        new_state = self.make_new_state(model=model, t=t + h, y=y_new)

        return new_state
