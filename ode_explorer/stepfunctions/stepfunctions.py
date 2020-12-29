from typing import Tuple

import numpy as np

from ode_explorer.models import ODEModel, HamiltonianSystem
from ode_explorer.stepfunctions.stepfunctions_impl import *
from ode_explorer.stepfunctions.templates import *
from ode_explorer.types import ModelState, StateVariable
from ode_explorer.utils.helpers import is_scalar

__all__ = ["ForwardEulerMethod",
           "HeunMethod",
           "RungeKutta4",
           "DOPRI45",
           "BackwardEulerMethod",
           "AdamsBashforth2",
           "BDF2",
           "EulerA",
           "EulerB"]


class ForwardEulerMethod(SingleStepMethod):
    """
    Forward Euler method for ODE integration.
    """

    def __init__(self):
        super(ForwardEulerMethod, self).__init__(order=1)

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        t, y = self.get_data_from_state(state=state)

        y_new = forward_euler_impl(model=model, t=t, y=y, h=h)

        return self.make_new_state(t=t + h, y=y_new)


class HeunMethod(SingleStepMethod):
    """
    Heun method for ODE integration.
    """

    def __init__(self):
        super(HeunMethod, self).__init__(order=2)
        self.num_stages = 2
        self.model_dim = 1
        self.k = np.zeros(self.num_stages)

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.k.shape:
            self._adjust_dims(y)

        y_new = heun_impl(model=model, t=t, y=y, h=h, k=self.k)

        return self.make_new_state(t=t + h, y=y_new)


class RungeKutta4(SingleStepMethod):
    """
    Classic Runge Kutta of order 4 for ODE integration.
    """

    def __init__(self):
        super(RungeKutta4, self).__init__(order=4)

        self.num_stages = 4
        self.k = np.zeros(self.num_stages)
        self.model_dim = 1

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.k.shape:
            self._adjust_dims(y)

        y_new = rk4_impl(model=model, t=t, y=y, h=h, k=self.k)

        return self.make_new_state(t=t + h, y=y_new)


class DOPRI45(SingleStepMethod):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with two y values, one accurate of order 4 and the other of order 5
    (hence the name), which can be used for step size estimation.
    """

    def __init__(self):
        super(DOPRI45, self).__init__(order=5)
        self.num_stages = 7
        self.k = np.zeros(self.num_stages)
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

        if self._get_shape(y) != self.k.shape:
            self._adjust_dims(y)

        y_new4, y_new5 = dopri45_impl(model=model, t=t, y=y, h=h, alphas=self.alphas,
                                      betas=self.betas, gammas=self.gammas, k=self.k)

        # 4th and 5th order solution
        new_state4 = self.make_new_state(t=t + h, y=y_new4)
        new_state5 = self.make_new_state(t=t + h, y=y_new5)

        return new_state4, new_state5


class BackwardEulerMethod(SingleStepMethod):
    """
    Implicit Euler Method for ODE solving.
    """

    def __init__(self, **kwargs):
        super(BackwardEulerMethod, self).__init__(order=2)

        self.num_stages = 1
        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:

        t, y = self.get_data_from_state(state=state)

        if is_scalar(y):
            y_new = backward_euler_scalar_impl(model=model, t=t, y=y, h=h, **self.solver_kwargs)
        else:
            y_new = backward_euler_ndim_impl(model=model, t=t, y=y, h=h, **self.solver_kwargs)

        return self.make_new_state(t=t + h, y=y_new)


class AdamsBashforth2(ExplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: SingleStepMethod):
        a_coeffs = np.ones(1)
        b_coeffs = np.array([1.5, -0.5])
        super(AdamsBashforth2, self).__init__(order=2,
                                              startup=startup,
                                              a_coeffs=a_coeffs,
                                              b_coeffs=b_coeffs)


class EulerA(SingleStepMethod):
    """
    EulerA method for Hamiltonian Systems integration.
    """

    def __init__(self):
        super(EulerA, self).__init__(order=1)

    @staticmethod
    def make_new_state(t: StateVariable, *state_vectors) -> ModelState:
        q, p = state_vectors
        return t, q, p

    def forward(self,
                hamiltonian: HamiltonianSystem,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        t, q, p = self.get_data_from_state(state=state)

        if not hamiltonian.is_separable:
            raise ValueError("EulerA for a non-separable Hamiltonian "
                             "is not implemented yet.")

        q_new, p_new = euler_a_separable_impl(hamiltonian=hamiltonian, t=t, q=q, p=p, h=h)

        return self.make_new_state(t + h, q_new, p_new)


class EulerB(SingleStepMethod):
    """
    EulerA method for Hamiltonian Systems integration.
    """

    def __init__(self):
        super(EulerB, self).__init__(order=1)

    @staticmethod
    def make_new_state(t: StateVariable, *state_vectors) -> ModelState:
        q, p = state_vectors
        return t, q, p

    def forward(self,
                hamiltonian: HamiltonianSystem,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        t, q, p = self.get_data_from_state(state=state)

        if not hamiltonian.is_separable:
            raise ValueError("EulerB for a non-separable Hamiltonian "
                             "is not implemented yet.")

        q_new, p_new = euler_b_separable_impl(hamiltonian=hamiltonian, t=t, q=q, p=p, h=h)

        return self.make_new_state(t + h, q_new, p_new)


class BDF2(ImplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: SingleStepMethod):
        a_coeffs = np.array([-4 / 3, 1 / 3])
        b_coeffs = np.array([2 / 3])
        super(BDF2, self).__init__(order=2,
                                   startup=startup,
                                   a_coeffs=a_coeffs,
                                   b_coeffs=b_coeffs)
