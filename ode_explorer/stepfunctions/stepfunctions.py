import jax.numpy as jnp

from ode_explorer.models import HamiltonianSystem
from ode_explorer.stepfunctions.stepfunctions_impl import *
from ode_explorer.stepfunctions.templates import *
from ode_explorer.types import ModelState

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
    order = 1
    _step = staticmethod(forward_euler_step)


class HeunMethod(SingleStepMethod):
    """
    Heun method for ODE integration.
    """
    order = 2
    _step = staticmethod(heun_step)


class RungeKutta4(SingleStepMethod):
    """
    Classic Runge Kutta of order 4 for ODE integration.
    """
    order = 4
    _step = staticmethod(rk4_step)


class DOPRI45(SingleStepMethod):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with two y values, one accurate of order 4 and the other of order 5
    (hence the name), which can be used for step size estimation.
    """
    order = 5
    _step = staticmethod(dopri45_step)


class BackwardEulerMethod(SingleStepMethod):
    """
    Implicit Euler Method for ODE solving.
    """
    order = 2
    _step = staticmethod(backward_euler_ndim_step)


class AdamsBashforth2(ExplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: SingleStepMethod):
        a_coeffs = jnp.ones(1)
        b_coeffs = jnp.array([1.5, -0.5])
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
    def make_new_state(t: jnp.ndarray, *state_vectors) -> ModelState:
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

        q_new, p_new = euler_a_separable_step(hamiltonian=hamiltonian, t=t, q=q, p=p, h=h)

        return self.make_new_state(t + h, q_new, p_new)


class EulerB(SingleStepMethod):
    """
    EulerA method for Hamiltonian Systems integration.
    """

    def __init__(self):
        super(EulerB, self).__init__(order=1)

    @staticmethod
    def make_new_state(t: jnp.ndarray, *state_vectors) -> ModelState:
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

        q_new, p_new = euler_b_separable_step(hamiltonian=hamiltonian, t=t, q=q, p=p, h=h)

        return self.make_new_state(t + h, q_new, p_new)


class BDF2(ImplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """

    def __init__(self, startup: SingleStepMethod):
        a_coeffs = jnp.array([-4 / 3, 1 / 3])
        b_coeffs = jnp.array([2 / 3])
        super(BDF2, self).__init__(order=2,
                                   startup=startup,
                                   a_coeffs=a_coeffs,
                                   b_coeffs=b_coeffs)
