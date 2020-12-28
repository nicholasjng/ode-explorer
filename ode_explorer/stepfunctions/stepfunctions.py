from typing import Tuple

import numpy as np
from scipy.optimize import root, root_scalar

from ode_explorer.models import ODEModel
from ode_explorer.stepfunctions.templates import SingleStepMethod, ExplicitMultiStepMethod
from ode_explorer.types import ModelState, StateVariable
from ode_explorer.utils.helpers import is_scalar
from ode_explorer.stepfunctions.stepfunc_impl import euler_scalar, euler_ndim

__all__ = ["EulerMethod",
           "EulerCython",
           "HeunMethod",
           "RungeKutta4",
           "DOPRI5",
           "DOPRI45",
           "ImplicitEulerMethod",
           "AdamsBashforth2"]


class EulerMethod(SingleStepMethod):
    """
    Euler method for ODE integration.
    """

    def __init__(self):
        super(EulerMethod, self).__init__()

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        y_new = y + h * model(t, y)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class EulerCython(SingleStepMethod):
    """
    Euler method for ODE integration.
    """

    def __init__(self):
        super(EulerCython, self).__init__()

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if is_scalar(y):
            y_new = euler_scalar(model.ode_fn, t, y, h, **kwargs)
        else:
            y_new = euler_ndim(model.ode_fn, t, y, h, **kwargs)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class HeunMethod(SingleStepMethod):
    """
    Heun method for ODE integration.
    """

    def __init__(self):
        super(HeunMethod, self).__init__(order=2)
        self.num_stages = 2
        self.model_dim = 1
        self.ks = np.zeros(self.num_stages)
        self.axis = None

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        hs = 0.5 * h
        ks = self.ks

        ks[0] = model(t, y)
        ks[1] = model(t + h, ks[0])
        y_new = y + hs * np.sum(ks, axis=self.axis)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class RungeKutta4(SingleStepMethod):
    """
    Classic Runge Kutta of order 4 for ODE integration.
    """

    def __init__(self):
        super(RungeKutta4, self).__init__(order=4)

        self.gammas = np.array([1.0, 2.0, 2.0, 1.0]) / 6
        self.num_stages = 4
        self.ks = np.zeros(self.num_stages)
        self.axis = None
        self.model_dim = 1

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        # notation follows that in
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        hs = 0.5 * h
        ks = self.ks

        ks[0] = model(t, y)
        ks[1] = model(t + hs, y + hs * ks[0])
        ks[2] = model(t + hs, y + hs * ks[1])
        ks[3] = model(t + h, y + h * ks[2])

        y_new = y + h * np.dot(self.gammas, ks)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class DOPRI5(SingleStepMethod):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with an approximation of order 5 in the step size.
    """

    def __init__(self):
        super(DOPRI5, self).__init__(order=5)
        self.num_stages = 6
        self.ks = np.zeros(self.num_stages)
        self.axis = None
        self.model_dim = 1

        # RK-specific variables
        self.alphas = np.array([0.2, 0.3, 0.8, 8 / 9, 1.0, 1.0])
        self.betas = [np.array([0.2]),
                      np.array([3 / 40, 9 / 40]),
                      np.array([44 / 45, -56 / 15, 32 / 9]),
                      np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
                      np.array([9017 / 3168, 355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656])]

        # First same as last (FSAL) rule
        self.gammas = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        hs = self.alphas * h
        ks = self.ks

        ks[0] = model(t, y)
        ks[1] = model(t + hs[0], y + h * np.dot(self.betas[0], ks[:1]))
        ks[2] = model(t + hs[1], y + h * np.dot(self.betas[1], ks[:2]))
        ks[3] = model(t + hs[2], y + h * np.dot(self.betas[2], ks[:3]))
        ks[4] = model(t + hs[3], y + h * np.dot(self.betas[3], ks[:4]))
        ks[5] = model(t + hs[4], y + h * np.dot(self.betas[4], ks[:5]))

        # 5th order solution, returned in 6 evaluations
        # step size estimation does not happen here
        y_new = y + h * np.dot(self.gammas, ks)

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class DOPRI45(SingleStepMethod):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with two y values, one accurate of order 4 and the other of order 5
    (hence the name), which can be used for step size estimation.
    """

    def __init__(self):
        super(DOPRI45, self).__init__(order=5)
        self.num_stages = 7
        self.ks = np.zeros(self.num_stages)
        self.axis = None
        self.model_dim = 1

        # RK-specific variables
        self.alphas = np.array([0.2, 0.3, 0.8, 8 / 9, 1.0, 1.0])
        self.betas = [np.array([0.2]),
                      np.array([3 / 40, 9 / 40]),
                      np.array([44 / 45, -56 / 15, 32 / 9]),
                      np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
                      np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
                      np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])]

        # First same as last (FSAL) rule
        self.gammas = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> Tuple[ModelState, ...]:

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.ks.shape:
            self._adjust_dims(y)

        hs = self.alphas * h
        ks = self.ks

        # FSAL rule, first eval is last eval of previous step
        ks[0] = model(t, y)
        ks[1] = model(t + hs[0], y + h * ks[0] * self.betas[0])
        ks[2] = model(t + hs[1], y + h * np.dot(self.betas[1], ks[:2]))
        ks[3] = model(t + hs[2], y + h * np.dot(self.betas[2], ks[:3]))
        ks[4] = model(t + hs[3], y + h * np.dot(self.betas[3], ks[:4]))
        ks[5] = model(t + hs[4], y + h * np.dot(self.betas[4], ks[:5]))

        # 5th order solution, computed in 6 evaluations
        y_new5 = y + h * np.dot(self.betas[-1], ks[:6])

        ks[6] = y_new5

        y_new4 = y + h * np.dot(self.gammas, ks)

        # 4th and 5th order solution
        new_state4 = self.make_new_state(t=t+h, y=y_new4)
        new_state5 = self.make_new_state(t=t+h, y=y_new5)

        return new_state4, new_state5


class ImplicitEulerMethod(SingleStepMethod):
    """
    Implicit Euler Method for ODE solving.
    """

    def __init__(self, **kwargs):
        super(ImplicitEulerMethod, self).__init__(order=2)

        # Runge-Kutta specific variables
        self.alpha = 1.0
        self.gamma = 1.0
        self.num_stages = 1

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        def F(x: StateVariable) -> StateVariable:
            # kwargs are not allowed in scipy.optimize, so pass tuple instead
            return y + h * model(t + h, x) - x

        # sort the kwargs before putting them into the tuple passed to root
        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        if is_scalar(y):
            root_res = root_scalar(F, args=args, x0=y, x1=y+h, **self.solver_kwargs)
            y_new = root_res.root
        else:
            root_res = root(F, x0=y, args=args, **self.solver_kwargs)
            y_new = root_res.x

        new_state = self.make_new_state(t=t+h, y=y_new)

        return new_state


class AdamsBashforth2(ExplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: SingleStepMethod):

        b_coeffs = np.array([1.5, -0.5])
        super(AdamsBashforth2, self).__init__(order=2,
                                              startup=startup,
                                              b_coeffs=b_coeffs)
