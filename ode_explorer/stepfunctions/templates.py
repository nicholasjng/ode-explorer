import logging

import numpy as np
from scipy.optimize import root, root_scalar

from ode_explorer.models import ODEModel
from ode_explorer.types import StateVariable, ModelState
from ode_explorer.utils.helpers import is_scalar

logger = logging.getLogger(__name__)


class SingleStepMethod:
    """
    Base class for all ODE step functions.
    """

    def __init__(self, order: int = 0):
        # order of the method
        self.order = order
        self.model_dim = 0
        self.num_stages = 0

    def _adjust_dims(self, y: StateVariable):
        scalar_ode = is_scalar(y)

        if scalar_ode:
            model_dim = 1
            shape = (self.num_stages,)
        else:
            model_dim = len(y)
            shape = (self.num_stages, model_dim)

        self.model_dim = model_dim
        self.ks = np.zeros(shape=shape)

    def _get_shape(self, y: StateVariable):
        return (self.num_stages,) if is_scalar(y) else (self.num_stages, len(y))

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


class MultiStepMethod:
    """
    Base class for explicit multi-step ODE solving methods.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 b_coeffs: np.ndarray,
                 order: int = 0):

        self.order = order

        # startup calculation variables, only for multi-step methods
        self.ready = False
        self.startup = startup

        # multi-step method variables
        self.a_coeffs = np.array([1.0])  # unused
        self.b_coeffs = b_coeffs

        self.num_previous = len(b_coeffs)
        # side cache for additional steps
        self.state_cache = [tuple()] * self.num_previous
        self.f_cache = np.zeros(self.num_previous)

    @staticmethod
    def get_data_from_state(state: ModelState):
        return state

    @staticmethod
    def make_new_state(t: StateVariable, y: StateVariable) -> ModelState:
        return t, y

    def _adjust_dims(self, y: StateVariable):
        scalar_ode = is_scalar(y)

        if scalar_ode:
            model_dim = 1
            shape = (self.num_previous,)
        else:
            model_dim = len(y)
            shape = (self.num_previous, model_dim)

        self.model_dim = model_dim
        self.f_cache = np.zeros(shape=shape)

    def _get_shape(self, y: StateVariable):
        return (self.num_previous,) if is_scalar(y) else (len(y), self.num_previous)

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

        if self._get_shape(y) != self.f_cache.shape:
            self._adjust_dims(y)

        self.state_cache[0] = state
        # fill function evaluation cache
        self.f_cache[0] = model(t, y)

        for i in range(1, self.num_previous):
            startup_state = self.startup.forward(model=model,
                                                 state=state,
                                                 h=h,
                                                 **kwargs)

            self.state_cache[i] = startup_state
            t1, y1 = self.get_data_from_state(state=startup_state)
            self.f_cache[i] = model(t1, y1)

        self.ready = True

    def get_cached_state(self, t: float):
        eps = 1e-15
        t_cache = np.array([state[0] for state in self.state_cache])
        closest_in_cache = np.isclose(t_cache, t)
        idx = np.argmax(closest_in_cache) + 1
        if not any(closest_in_cache):
            idx = np.argmax(t_cache > t + eps)

        return self.state_cache[idx]

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        raise NotImplementedError


class ExplicitRungeKuttaMethod(SingleStepMethod):
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
        hb = self.betas * h
        ks = self.ks

        ks[0] = model(t, y)

        for i in range(1, self.num_stages):
            # first row of betas is a zero row because it is an explicit RK
            ks[i] = model(t + ha[i], y + np.dot(hb[i], ks))

        y_new = y + np.dot(hg, ks)

        new_state = self.make_new_state(t=t + h, y=y_new)

        return new_state


class ImplicitRungeKuttaMethod(SingleStepMethod):
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

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        ha = self.alphas * h
        hb = self.betas * h
        hg = self.gammas * h
        ks = self.ks

        initial_shape = ks.shape
        shape_prod = np.prod(initial_shape)

        def F(x: np.ndarray) -> np.ndarray:

            # kwargs are not allowed in scipy.optimize, so pass tuple instead
            model_stack = np.hstack(model(t + ha[i], np.dot(hb[i], x.reshape(initial_shape)))
                                    for i in range(self.num_stages))

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

        if shape_prod != 1:
            # TODO: Retry here in case of convergence failure?
            root_res = root(F, x0=ks.reshape((shape_prod,)), args=args, **self.solver_kwargs)

            y_new = y + np.dot(hg, root_res.x.reshape(initial_shape))

        else:
            root_res = root_scalar(F_scalar, x0=y, x1=y + hg[0], args=args, **self.solver_kwargs)

            y_new = y + hg[0] * root_res.root

        new_state = self.make_new_state(t=t + h, y=y_new)

        return new_state


class ExplicitMultiStepMethod(MultiStepMethod):
    """
    Base class for explicit multi-step ODE solving methods.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 b_coeffs: np.ndarray,
                 order: int = 0):

        super(ExplicitMultiStepMethod, self).__init__(startup=startup,
                                                      b_coeffs=b_coeffs,
                                                      order=order)

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

            return self.state_cache[0]

        t, y = self.get_data_from_state(state=state)

        # This branch is curious
        eps = 1e-12
        # TODO: fix this
        t_cache = np.array([state[0] for state in self.state_cache])
        if t + eps < t_cache[-1]:
            return self.get_cached_state(t)

        y_new = y + h * np.dot(self.b_coeffs, self.f_cache)

        # shift all states and all f evaluations to the left by 1
        self.state_cache.pop(0)
        self.state_cache.append(self.make_new_state(t=t + h, y=y_new))

        self.f_cache = np.roll(self.f_cache, shift=-1, axis=0)
        self.f_cache[-1] = model(t + h, y_new)

        return self.state_cache[-1]


class ImplicitMultiStepMethod(MultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 b_coeffs: np.ndarray,
                 order: int = 0,
                 **kwargs):

        super(ImplicitMultiStepMethod, self).__init__(startup=startup,
                                                      b_coeffs=b_coeffs,
                                                      order=order)

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        if not self.ready:
            # startup calculation to the multi-step method,
            # fills the state and f-caches
            self.perform_startup_calculation(model=model,
                                             state=state,
                                             h=h,
                                             **kwargs)

            # first cached value
            return self.state_cache[0]

        t, y = self.get_data_from_state(state=state)

        # This branch is curious
        eps = 1e-12
        # TODO: fix this
        t_cache = np.array(state[0] for state in self.state_cache)
        if t + eps < t_cache[-1]:
            return self.get_cached_state(t)

        def F(x: StateVariable) -> StateVariable:
            return h * (model(t + h, x) + y - np.dot(self.b_coeffs, self.f_cache)) - x

        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        root_res = root(F, x0=self.f_cache[-1], args=args, **self.solver_kwargs)

        y_new = root_res.x

        # shift all states and all f evaluations to the left by 1,
        self.state_cache.pop(0)
        self.state_cache.append(self.make_new_state(t=t + h, y=y_new))

        self.f_cache = np.roll(self.f_cache, shift=-1, axis=0)
        self.f_cache[-1] = model(t + h, y_new)

        return self.state_cache[-1]
