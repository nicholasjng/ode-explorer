import numpy as np
import inspect
from typing import Dict, Text, Any, Union
from ode_explorer.model import ODEModel
from ode_explorer.constants import ZIPPED, VARIABLES
from utils.data_utils import is_scalar

from scipy.optimize import root, root_scalar
# import jax.numpy as jnp
# from jax import grad, jit, vmap

__all__ = ["StepFunction",
           "EulerMethod",
           "HeunMethod",
           "RungeKutta4",
           "DOPRI5",
           "DOPRI45",
           "ImplicitEulerMethod",
           "AdamsBashforth2"]


class StepFunction:
    """
    Base class for all ODE step functions.
    """

    def __init__(self):
        # order of the method
        self.order = 0

    @staticmethod
    def get_data_from_state_dict(model: ODEModel,
                                 state_dict: Dict[Text, float]):
        # TODO here: Pass format argument for smart format inference
        t = state_dict.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        if all(var in state_dict for var in model.variable_names):
            y = state_dict[model.variable_names[0]]
        else:
            y = np.array(list(state_dict.values()))

        # scalar ODE, return just the value then
        if not is_scalar(y) and len(y) == 1:
            return t, y[0]

        return t, y

    @staticmethod
    def make_zipped_dict(model: ODEModel, t: float, y: Any):
        if is_scalar(y):
            y_new = [y]
        else:
            y_new = y

        return {**{model.indep_name: t},
                **dict(zip(model.dim_names, y_new))}

    @staticmethod
    def make_state_dict(model: ODEModel, t: float, y: Any):
        return {model.indep_name: t, model.variable_names[0]: y}

    def forward(self,
                model: ODEModel,
                state: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        raise NotImplementedError


class EulerMethod(StepFunction):
    """
    Euler method for ODE integration.
    """

    def __init__(self):
        super().__init__()
        self.order = 1

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        y_new = y + h * model(t, y, **kwargs)

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state


class HeunMethod(StepFunction):
    """
    Heun method for ODE integration.
    """

    def __init__(self):
        super(HeunMethod, self).__init__()
        self.order = 1
        self.ks = np.zeros((1, 2))

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        if hasattr(y, "__len__") and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), 4))

        hs = 0.5 * h
        ks = self.ks

        ks[:, 0] = model(t, y, **kwargs)
        ks[:, 1] = model(t + h, ks[:, 0], **kwargs)
        y_new = y + hs * np.sum(ks, axis=1)

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state


class RungeKutta4(StepFunction):
    """
    Classic Runge Kutta of order 4 for ODE integration.
    """

    def __init__(self):
        super(RungeKutta4, self).__init__()

        self.order = 4
        self.gammas = np.array([1.0, 2.0, 2.0, 1.0]) / 6
        self.ks = np.zeros((1, 4))

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        if hasattr(y, "__len__") and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), 4))

        # notation follows that in
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        hs = 0.5 * h
        ks = self.ks

        ks[:, 0] = model(t, y, **kwargs)
        ks[:, 1] = model(t + hs, y + hs * ks[:, 0], **kwargs)
        ks[:, 2] = model(t + hs, y + hs * ks[:, 1], **kwargs)
        ks[:, 3] = model(t + h, y + h * ks[:, 2], **kwargs)

        y_new = y + h * np.sum(ks * self.gammas, axis=1)

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state


class DOPRI5(StepFunction):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with an approximation of order 5 in the step size.
    """

    def __init__(self):
        super(DOPRI5, self).__init__()
        self.order = 5
        self.ks = np.zeros((1, 6))

        # RK-specific variables
        self.alphas = np.array([0.2, 0.3, 0.8, 8 / 9, 1.0, 1.0])
        self.betas = [np.array([0.2]),
                      np.array([3 / 40, 9 / 40]),
                      np.array([44 / 45, -56 / 15, 32 / 9]),
                      np.array([19372 / 6561, -25360 / 2187, 64448 / 6561,
                                -212 / 729]),
                      np.array([9017 / 3168, 355 / 33, 46732 / 5247, 49 / 176,
                                -5103 / 18656])]

        # First same as last (FSAL) rule
        self.gammas = np.array(
            [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        if hasattr(y, "__len__") and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), 6))

        hs = self.alphas * h
        ks = self.ks

        # FSAL rule, first eval is last eval of previous step
        ks[:, 0] = model(t, y, **kwargs)
        ks[:, 1] = model(t + hs[0], y + h * ks[:, 0] * self.betas[0], **kwargs)
        ks[:, 2] = model(t + hs[1],
                         y + h * np.sum(ks[:, :2] * self.betas[1], axis=1),
                         **kwargs)
        ks[:, 3] = model(t + hs[2],
                         y + h * np.sum(ks[:, :3] * self.betas[2], axis=1),
                         **kwargs)
        ks[:, 4] = model(t + hs[3],
                         y + h * np.sum(ks[:, :4] * self.betas[3], axis=1),
                         **kwargs)
        ks[:, 5] = model(t + hs[4],
                         y + h * np.sum(ks[:, :5] * self.betas[4], axis=1),
                         **kwargs)

        # 5th order solution, returned in 6 evaluations
        # step size estimation does not happen here
        y_new = y + h * np.sum(ks * self.gammas, axis=1)

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state


class DOPRI45(StepFunction):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with two y values, one accurate of order 4 and the other of order 5
    (hence the name), which can be used for step size estimation.
    """
    def __init__(self):
        super(DOPRI45, self).__init__()
        self.order = 5

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:
        pass


class ImplicitEulerMethod(StepFunction):
    """
    Implicit Euler Method for ODE solving.
    """
    def __init__(self, **kwargs):
        super(ImplicitEulerMethod, self).__init__()
        self.order = 1

        # Runge-Kutta specific variables
        self.k = np.ones(1)
        self.alpha = 1.0
        self.gamma = 1.0

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        if hasattr(y, "__len__") and len(y) != len(self.k):
            self.k = np.zeros(len(y))

        def F(x, *args) -> Union[np.ndarray, float]:
            # kwargs are not allowed in scipy.optimize.root, so
            # pass tuple instead
            return y + h * model(t + h, x, *args) - x

        # this bit is important to sort the kwargs before putting them into
        # the tuple passed to root
        model_spec = inspect.getfullargspec(model.ode_fn).args[2:]
        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        if is_scalar(y):
            root_res = root_scalar(F, args=args, **self.solver_kwargs)
        else:
            root_res = root(F, x0=self.k, args=args, **self.solver_kwargs)

        y_new = root_res.x

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state


class AdamsBashforth2(StepFunction):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: StepFunction):
        super(AdamsBashforth2, self).__init__()
        self.order = 2

        # startup calculation variables, only for multistep methods
        self.ready = False
        self.startup = startup
        # side cache for additional steps
        self.y_cache = np.ones((1, 2))
        self.t_cache = np.zeros(2)
        self.f_cache = np.zeros_like(self.y_cache)

        # multistep method variables
        self.a_coeffs = np.array([1.0])  # unused
        self.b_coeffs = np.array([1.5, -0.5])

    def reset(self):
        # Resets the run so that next time the step function is called,
        # new startup values will be calculated with the saved startup step
        # function. Useful if the step function is supposed to be reused in
        # multiple non-consecutive runs.
        self.ready = False

    def perform_startup_calculation(self, model: ODEModel,
                                    state_dict: Dict[Text,
                                                     Union[np.ndarray, float]],
                                    h: float,
                                    **kwargs):

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        if hasattr(y, "__len__") and len(y) != self.y_cache.shape[0]:
            self.y_cache = np.zeros((len(y), 2))

        self.t_cache[0], self.y_cache[:, 0] = t, y
        startup_state = self.startup.forward(model=model,
                                             state_dict=state_dict,
                                             h=h, **kwargs)

        t1, y1 = self.get_data_from_state_dict(model=model,
                                               state_dict=startup_state)

        self.t_cache[1], self.y_cache[:, 1] = t1, y1

        # fill function evaluation cache
        self.f_cache[:, 0] = model(t, y, **kwargs)
        self.f_cache[:, 1] = model(t1, y1, **kwargs)

        self.ready = True

    def forward(self,
                model: ODEModel,
                state_dict: Dict[Text, Union[np.ndarray, float]],
                h: float,
                return_format: Text = "variables",
                **kwargs) -> Dict[Text, Union[np.ndarray, float]]:

        if not self.ready:
            # startup calculation to the multistep method,
            # fills the y-, t- and f-caches
            self.perform_startup_calculation(model=model,
                                             state_dict=state_dict, h=h,
                                             **kwargs)

            y_new = self.y_cache[:, -1]

            if return_format == "zipped":
                new_state = self.make_zipped_dict(model=model,
                                                  t=self.t_cache[-1],
                                                  y=y_new)
            else:
                new_state = self.make_state_dict(model=model,
                                                 t=self.t_cache[-1],
                                                 y=y_new)

            return new_state

        t, y = self.get_data_from_state_dict(model=model,
                                             state_dict=state_dict)

        # TODO: Decide here whether to return cached values based on
        #  the condition t <= self.t_cache[-1]. Return the last cached value
        #  with t < self.t_cache[i]. This is O(n), but n is small
        #  (order of method) so what gives

        y_new = y + h * np.sum(self.b_coeffs * self.f_cache, axis=1)

        # shift all y and all f evaluations to the left by 1,
        # we only need the two previous steps
        self.y_cache = np.roll(self.y_cache, shift=-1, axis=1)
        self.f_cache = np.roll(self.f_cache, shift=-1, axis=1)

        self.y_cache[:, -1] = y_new
        self.f_cache[:, -1] = model(t + h, y_new, **kwargs)

        if return_format == "zipped":
            new_state = self.make_zipped_dict(model=model, t=t+h, y=y_new)
        else:
            new_state = self.make_state_dict(model=model, t=t+h, y=y_new)

        return new_state
