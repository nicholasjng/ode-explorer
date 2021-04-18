import jax.numpy as jnp

from ode_explorer.stepfunctions.stepfunctions_impl import *
from ode_explorer.stepfunctions.templates import *

__all__ = ["ForwardEulerMethod",
           "HeunMethod",
           "RungeKutta4",
           "DOPRI45",
           # "BackwardEulerMethod",
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


# class BackwardEulerMethod(SingleStepMethod):
#     """
#     Implicit Euler Method for ODE solving.
#     """
#     order = 2
#     _step = staticmethod(backward_euler_ndim_step)


class AdamsBashforth2(ExplicitMultiStepMethod):
    """
    Adams-Bashforth Method of order 2 for ODE solving.
    """
    order = 2
    a_coeffs = jnp.ones(1)
    b_coeffs = jnp.array([1.5, -0.5])


class BDF2(ImplicitMultiStepMethod):
    """
    Backward Differentiation Formula of order 2 for ODE solving.
    """
    order = 2
    a_coeffs = jnp.array([-4 / 3, 1 / 3])
    b_coeffs = jnp.array([2 / 3])


class EulerA(SingleStepMethod):
    """
    EulerA method for Hamiltonian Systems integration.
    """
    order = 1
    _step = staticmethod(euler_a_step)

    @staticmethod
    def make_new_state(t: jnp.ndarray, *state_vectors):
        q, p = state_vectors
        return t, q, p


class EulerB(SingleStepMethod):
    """
    EulerB method for Hamiltonian Systems integration.
    """
    order = 1
    _step = staticmethod(euler_b_step)

    @staticmethod
    def make_new_state(t: jnp.ndarray, *state_vectors):
        q, p = state_vectors
        return t, q, p
