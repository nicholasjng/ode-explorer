from typing import List

import numpy as np
from scipy.optimize import root, root_scalar

from ode_explorer.models import ODEModel, HamiltonianSystem
from ode_explorer.types import ModelState, StateVariable

__all__ = ["forward_euler_impl",
           "heun_impl",
           "rk4_impl",
           "dopri45_impl",
           "backward_euler_scalar_impl",
           "backward_euler_ndim_impl",
           "euler_a_separable_impl",
           "euler_b_separable_impl"]


def forward_euler_impl(model: ODEModel, t: StateVariable, y: StateVariable, h: float) -> StateVariable:
    return y + h * model(t, y)


def heun_impl(model: ODEModel, t: StateVariable, y: StateVariable, h: float, k: np.ndarray) -> StateVariable:
    hs = np.ones(2) * 0.5 * h

    k[0] = model(t, y)
    k[1] = model(t + h, y + h * k[0])
    return y + np.dot(hs, k)


def rk4_impl(model: ODEModel, t: StateVariable, y: StateVariable, h: float, k: np.ndarray) -> StateVariable:
    # notation follows that in
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    hs = 0.5 * h
    gammas = np.array([1.0, 2.0, 2.0, 1.0]) / 6

    k[0] = model(t, y)
    k[1] = model(t + hs, y + hs * k[0])
    k[2] = model(t + hs, y + hs * k[1])
    k[3] = model(t + h, y + h * k[2])

    return y + h * np.dot(gammas, k)


def dopri45_impl(model: ODEModel, t: StateVariable, y: StateVariable, h: float, alphas: np.ndarray,
                 betas: List[np.ndarray], gammas: np.ndarray, k: np.ndarray) -> ModelState:
    k[0] = model(t, y)
    k[1] = model(t + h * alphas[0], y + h * np.dot(betas[0], k[:1]))
    k[2] = model(t + h * alphas[1], y + h * np.dot(betas[1], k[:2]))
    k[3] = model(t + h * alphas[2], y + h * np.dot(betas[2], k[:3]))
    k[4] = model(t + h * alphas[3], y + h * np.dot(betas[3], k[:4]))
    k[5] = model(t + h * alphas[4], y + h * np.dot(betas[4], k[:5]))

    # 5th order solution, computed in 6 evaluations
    y_new5 = y + h * np.dot(betas[5], k[:6])

    k[6] = y_new5

    # 4th order solution, to be used in error estimation
    y_new4 = y + h * np.dot(gammas, k)

    return y_new4, y_new5


def backward_euler_scalar_impl(model: ODEModel, t: StateVariable, y: float, h: float,
                               **solver_kwargs) -> float:
    def F(x: float) -> float:
        return y + h * model(t + h, x) - x

    # sort the kwargs before putting them into the tuple passed to root
    # if kwargs:
    #     args = tuple(kwargs[arg] for arg in model.fn_args.keys())
    # else:
    #     args = ()
    args = ()

    # TODO: Retry here in case of convergence failure?
    root_res = root_scalar(F, args=args, x0=y, x1=y + h, **solver_kwargs)
    y_new = root_res.root

    return y_new


def backward_euler_ndim_impl(model: ODEModel, t: StateVariable, y: StateVariable, h: float,
                             **solver_kwargs) -> StateVariable:
    def F(x: StateVariable) -> StateVariable:
        return y + h * model(t + h, x) - x

    # sort the kwargs before putting them into the tuple passed to root
    # if kwargs:
    #     args = tuple(kwargs[arg] for arg in model.fn_args.keys())
    # else:
    #     args = ()
    args = ()

    # TODO: Retry here in case of convergence failure?
    root_res = root(F, x0=y, args=args, **solver_kwargs)
    y_new = root_res.x

    return y_new


def euler_a_separable_impl(hamiltonian: HamiltonianSystem, t: StateVariable, q: StateVariable,
                           p: StateVariable, h: float) -> ModelState:
    q_new = q + h * hamiltonian.p_derivative(t, p)
    p_new = p - h * hamiltonian.q_derivative(t, q_new)

    return q_new, p_new


def euler_b_separable_impl(hamiltonian: HamiltonianSystem, t: StateVariable, q: StateVariable,
                           p: StateVariable, h: float) -> ModelState:
    p_new = p - h * hamiltonian.q_derivative(t, q)
    q_new = q + h * hamiltonian.p_derivative(t, p_new)

    return q_new, p_new
