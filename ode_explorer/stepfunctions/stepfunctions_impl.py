from jax import lax
import jax.numpy as jnp
# from scipy.optimize import root, root_scalar

from ode_explorer.models import ODEModel, HamiltonianSystem
from ode_explorer.types import ModelState

__all__ = ["forward_euler_step",
           "heun_step",
           "rk4_step",
           "dopri45_step",
           # "backward_euler_scalar_step",
           # "backward_euler_ndim_step",
           "euler_a_step",
           "euler_b_step"]


def forward_euler_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array) -> jnp.array:
    return y + h * model(t, y)


def heun_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array) -> jnp.array:

    k1 = h * model(t, y)
    k2 = h * model(t + h, y + h * k1)
    return y + 0.5 * (k1 + k2)


def rk4_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array) -> jnp.array:
    # notation follows that in
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    hs = 0.5 * h

    k1 = h * model(t, y)
    k2 = h * model(t + hs, y + hs * k1)
    k3 = h * model(t + hs, y + hs * k2)
    k4 = h * model(t + h, y + h * k3)

    return y + 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def dopri45_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array) -> ModelState:

    # initialize k-buffer for intermediate values
    k = jnp.zeros((7, y.shape[0]), y.dtype).at[0].set(model(t, y))

    # RK-specific variables
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0])
    beta = jnp.array([
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
    ])
    gamma1 = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    # First same as last (FSAL) rule
    gamma2 = jnp.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

    def body_fun(i, k_buf):
        ti = t + h * alpha[i - 1]
        yi = y + h * jnp.dot(beta[i - 1], k_buf)
        fi = model(ti, yi)
        return k_buf.at[i].set(fi)

    k = lax.fori_loop(lower=1, upper=7, body_fun=body_fun, init_val=k)

    # 5th order solution, computed in 6 evaluations
    y1 = y + h * jnp.dot(gamma1, k)

    # 4th order solution, to be used in error estimation
    y2 = y + h * jnp.dot(gamma2, k)

    return y1, y2


# def backward_euler_scalar_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array,
#                                **solver_kwargs) -> jnp.array:
#     def F(x: jnp.array) -> jnp.array:
#         return y + h * model(t + h, x) - x
#
#     # sort the kwargs before putting them into the tuple passed to root
#     # if kwargs:
#     #     args = tuple(kwargs[arg] for arg in model.fn_args.keys())
#     # else:
#     #     args = ()
#     args = ()
#
#     root_res = root_scalar(F, args=args, x0=y, x1=y + h, **solver_kwargs)
#     y_new = root_res.root
#
#     return y_new


# def backward_euler_ndim_step(model: ODEModel, t: jnp.array, y: jnp.array, h: jnp.array,
#                              **solver_kwargs) -> jnp.array:
#     def F(x: jnp.array) -> jnp.array:
#         return y + h * model(t + h, x) - x
#
#     # sort the kwargs before putting them into the tuple passed to root
#     # if kwargs:
#     #     args = tuple(kwargs[arg] for arg in model.fn_args.keys())
#     # else:
#     #     args = ()
#     args = ()
#
#     root_res = root(F, x0=y, args=args, **solver_kwargs)
#     y_new = root_res.x
#
#     return y_new


def euler_a_step(hamiltonian: HamiltonianSystem, t: jnp.array, q: jnp.array, p: jnp.array, h: jnp.array) -> ModelState:
    q_new = q + h * hamiltonian.p_derivative(t, p)
    p_new = p - h * hamiltonian.q_derivative(t, q_new)

    return q_new, p_new


def euler_b_step(hamiltonian: HamiltonianSystem, t: jnp.array, q: jnp.array, p: jnp.array, h: jnp.array) -> ModelState:
    p_new = p - h * hamiltonian.q_derivative(t, q)
    q_new = q + h * hamiltonian.p_derivative(t, p_new)

    return q_new, p_new
