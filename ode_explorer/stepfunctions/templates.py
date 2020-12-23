import copy
import logging

import numpy as np
from scipy.optimize import root, root_scalar

from ode_explorer.models.model import ODEModel
from ode_explorer.types import StateVariable, ModelState
from ode_explorer.utils.helpers import is_scalar

logger = logging.getLogger(__name__)


class StepFunction:
    """
    Base class for all ODE step functions.
    """

    def __init__(self, order: int = 0):
        # order of the method
        self.order = order
        self.model_dim = 0
        self.num_stages = 0
        self.axis = None

    def _adjust_dims(self, y: StateVariable):
        scalar_ode = is_scalar(y)

        if scalar_ode:
            model_dim, axis = 1, None
            shape = (self.num_stages,)
        else:
            model_dim, axis = len(y), 0
            shape = (self.num_stages, model_dim)

        self.model_dim = model_dim
        self.axis = axis
        self.ks = np.zeros(shape=shape)

    def _get_shape(self, y: StateVariable):
        return (self.num_stages,) if is_scalar(y) else (len(y), self.num_stages)

    @staticmethod
    def get_data_from_state(state: ModelState):
        return state

    @staticmethod
    def make_new_state(t: StateVariable, y: StateVariable) -> ModelState:
        return t, y

    def forward(self,
                model: ODEModel,
                state_dict: ModelState,
                h: float,
                **kwargs) -> ModelState:
        raise NotImplementedError


class ExplicitRungeKuttaMethod(StepFunction):
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 order: int = 0):

        super(ExplicitRungeKuttaMethod, self).__init__(order=order)

        self.validate_butcher_tableau(alphas=alphas, betas=betas, gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.num_stages = len(self.alphas)
        self.ks = np.zeros(betas.shape[0])

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
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        ha = self.alphas * h
        hg = self.gammas * h
        ks = self.ks

        ks[0] = model(t, y)

        for i in range(1, self.num_stages):
            # first row of betas is a zero row because it is an explicit RK
            ks[i] = model(t + ha[i], y + ha[i] * np.dot(self.betas[i], ks[:i]))

        y_new = y + np.dot(hg, ks)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class ImplicitRungeKuttaMethod(StepFunction):
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 order: int = 0,
                 **kwargs):

        super(ImplicitRungeKuttaMethod, self).__init__(order=order)

        self.validate_butcher_tableau(alphas=alphas, betas=betas, gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.num_stages = len(self.alphas)
        self.ks = np.zeros(betas.shape[0])

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
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if not is_scalar(y) and len(y) != self.ks.shape[0]:
            self.ks = np.zeros((len(y), self.num_stages))

        ha = self.alphas * h
        hb = self.betas * h
        hg = self.gammas * h
        ks = self.ks

        n, m = ks.shape

        def F(x: np.ndarray) -> np.ndarray:

            # kwargs are not allowed in scipy.optimize, so pass tuple instead
            model_stack = np.hstack(model(t + ha[i], x.reshape((n, m)).dot(hb[i])) for i in range(m))

            return model_stack - x

        # modified function in case of using Implicit Euler method or
        # equivalents on a scalar ODE
        def F_scalar(x: float) -> float:
            return model(t + ha[0], y + hb[0] * x) - x

        # sort the kwargs before putting them into the tuple passed to root
        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        if n * m != 1:
            # TODO: Retry here in case of convergence failure?
            root_res = root(F, x0=ks.reshape((n * m,)), args=args, **self.solver_kwargs)

            y_new = y + np.dot(hg, root_res.x.reshape((n, m)))

        else:
            root_res = root_scalar(F_scalar, x0=y, x1=y+hg[0], args=args, **self.solver_kwargs)

            y_new = y + hg[0] * root_res.root

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class ExplicitMultiStepMethod(StepFunction):
    """
        Adams-Bashforth Method of order 2 for ODE solving.
        """

    def __init__(self,
                 startup: StepFunction,
                 b_coeffs: np.ndarray,
                 order: int = 0):

        super(ExplicitMultiStepMethod, self).__init__(order=order)

        # startup calculation variables, only for multi-step methods
        self.ready = False
        self.startup = startup

        # multi-step method variables
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

    def perform_startup_calculation(self,
                                    model: ODEModel,
                                    state: ModelState,
                                    h: float,
                                    **kwargs):

        t, y = self.get_data_from_state(state=state)

        if not is_scalar(y) and len(y) != self.y_cache.shape[0]:
            self.y_cache = np.zeros((len(y), self.lookback))

        self.t_cache[0], self.y_cache[:, 0] = t, y
        # fill function evaluation cache
        self.f_cache[:, 0] = model(t, y)

        dummy_state = copy.deepcopy(state)
        for i in range(1, self.lookback):
            startup_state = self.startup.forward(model=model,
                                                 state=dummy_state,
                                                 h=h,
                                                 **kwargs)

            t1, y1 = startup_state

            self.t_cache[i], self.y_cache[:, i] = t1, y1
            self.f_cache[:, i] = model(t1, y1)
            dummy_state = startup_state

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
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        if not self.ready:
            # startup calculation to the multi-step method,
            # fills the y-, t- and f-caches
            self.perform_startup_calculation(model=model,
                                             state=state,
                                             h=h,
                                             **kwargs)

            # first cached value
            y_new = self.y_cache[:, 1]

            new_state = self.make_new_state(t=self.t_cache[1],
                                            y=y_new)

            return new_state

        t, y = self.get_data_from_state(state=state)

        # This branch is curious
        eps = 1e-12
        if t + eps < self.t_cache[-1]:
            t_new, y_new = self.get_cached_values(t)

            new_state = self.make_new_state(t=t_new, y=y_new)

            return new_state

        y_new = y + h * np.sum(self.b_coeffs * self.f_cache, axis=1)

        # shift all y and all f evaluations to the left by 1,
        # we only need the two previous steps
        self.y_cache = np.roll(self.y_cache, shift=-1, axis=1)
        self.f_cache = np.roll(self.f_cache, shift=-1, axis=1)

        self.y_cache[:, -1] = y_new
        self.f_cache[:, -1] = model(t + h, y_new)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class ImplicitMultiStepMethod(StepFunction):
    """
        Adams-Bashforth Method of order 2 for ODE solving.
        """

    def __init__(self,
                 startup: StepFunction,
                 b_coeffs: np.ndarray,
                 order: int = 0,
                 **kwargs):
        super(ImplicitMultiStepMethod, self).__init__(order=order)
        # startup calculation variables, only for multi-step methods
        self.ready = False
        self.startup = startup

        # multi-step method variables
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
                                    state: ModelState,
                                    h: float,
                                    **kwargs):

        t, y = self.get_data_from_state(state=state)

        if not is_scalar(y) and len(y) != self.y_cache.shape[0]:
            self.y_cache = np.zeros((len(y), self.lookback))

        self.t_cache[0], self.y_cache[:, 0] = t, y
        # fill function evaluation cache
        self.f_cache[:, 0] = model(t, y)

        dummy_state = copy.deepcopy(state)
        for i in range(1, self.lookback):
            startup_state = self.startup.forward(model=model,
                                                 state=dummy_state,
                                                 h=h, **kwargs)

            t1, y1 = startup_state

            self.t_cache[i], self.y_cache[:, i] = t1, y1
            self.f_cache[:, i] = model(t1, y1)
            dummy_state = startup_state

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
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        if not self.ready:
            # startup calculation to the multi-step method,
            # fills the y-, t- and f-caches
            self.perform_startup_calculation(model=model,
                                             state=state,
                                             h=h,
                                             **kwargs)

            # first cached value
            y_new = self.y_cache[:, 1]

            new_state = self.make_new_state(t=self.t_cache[1], y=y_new)

            return new_state

        t, y = state

        # This branch is curious
        eps = 1e-12
        if t + eps < self.t_cache[-1]:
            t_new, y_new = self.get_cached_values(t)

            new_state = self.make_new_state(t=t_new, y=y_new)

            return new_state

        def F(x: StateVariable) -> StateVariable:
            return h * model(t + h, x) + y - \
                   h * np.sum(self.b_coeffs * self.f_cache, axis=1) - x

        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        root_res = root(F, x0=self.f_cache[:, -1], args=args, **self.solver_kwargs)

        y_new = root_res.x

        # shift all y and all f evaluations to the left by 1,
        # we only need the two previous steps
        self.y_cache = np.roll(self.y_cache, shift=-1, axis=1)
        self.f_cache = np.roll(self.f_cache, shift=-1, axis=1)

        self.y_cache[:, -1] = y_new
        self.f_cache[:, -1] = model(t + h, y_new)

        new_state = self.make_new_state(t=t + h, y=y_new)

        return new_state
