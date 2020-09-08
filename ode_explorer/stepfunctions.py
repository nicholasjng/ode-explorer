import numpy as np
from typing import Dict, Text, Any
from ode_explorer.model import ODEModel
from scipy.optimize import fsolve
import jax.numpy as jnp
from jax import grad, jit, vmap

__all__ = ["EulerMethod", "HeunMethod", "RungeKutta4", "DOPRI5"]


class StepFunction:
    """
    Base class for all ODE step functions.
    """
    def __init__(self):

        # order of the method
        self.order = 0

    @staticmethod
    def get_data_from_state_dict(model: ODEModel,
                                 state: Dict[Text, float]):

        t = state.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        y = np.array(list(state.values()))

        return t, y

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:
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
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:

        # copy here, might be expensive
        state_copy = state.copy()

        t, y = self.get_data_from_state_dict(model=model, state=state_copy)

        y_new = y + h * model(t, y, **kwargs)

        new_state = {**{model.indep_name: t + h},
                     **dict(zip(model.variable_names, y_new))}

        return new_state


class HeunMethod(StepFunction):
    """
    Heun method for ODE integration.
    """
    def __init__(self):
        super().__init__()
        self.order = 1

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:

        t, y = self.get_data_from_state_dict(model=model, state=state)

        hs = 0.5 * h
        k1 = model(t, y, **kwargs)
        y_new = y + hs * (k1 + model(t + h, k1))

        new_state = {**{model.indep_name: t + h},
                     **dict(zip(model.variable_names, y_new))}

        return new_state


class RungeKutta4(StepFunction):
    """
    Classic Runge Kutta of order 4 for ODE integration.
    """
    def __init__(self):
        super(RungeKutta4, self).__init__()

        self.order = 4
        self.gammas = np.array([1.0, 2.0, 2.0, 1.0]) / 6

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:

        # copy here, might be expensive
        #state_copy = state.copy()

        t, y = self.get_data_from_state_dict(model=model, state=state)

        # notation follows that in
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        hs = 0.5 * h
        # possible hotspot for large ODE systems
        ks = np.zeros((len(y), 4))
        ks[:, 0] = model(t, y, **kwargs)
        ks[:, 1] = model(t + hs, y + hs * ks[:, 0], **kwargs)
        ks[:, 2] = model(t + hs, y + hs * ks[:, 1], **kwargs)
        ks[:, 3] = model(t + h, y + h * ks[:, 2], **kwargs)

        y_new = y + h * np.sum(ks * self.gammas, axis=1)

        new_state = {**{model.indep_name: t + h},
                     **dict(zip(model.variable_names, y_new))}

        return new_state


class DOPRI5(StepFunction):
    """
    Dormand-Prince method for explicit ODE integration. This method returns a
    dict with two y values, one accurate of order 4 and the other of order 5
    (hence the name), which can be used for step size estimation.
    """
    def __init__(self):
        super(DOPRI5, self).__init__()
        self.order = 5
        self.alphas = np.array([0.2, 0.3, 0.8, 8/9, 1.0, 1.0])
        self.betas = [np.array([0.2]),
                      np.array([3/40, 9/40]),
                      np.array([44/45, -56/15, 32/9]),
                      np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
                      np.array([9017/3168, 355/33, 46732/5247, 49/176, -5103/18656])]

        # First same as last (FSAL) rule
        self.gammas = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:

        t, y = self.get_data_from_state_dict(model=model, state=state)

        hs = self.alphas * h
        # possible hotspot for large ODE systems
        ks = np.zeros((len(y), len(self.gammas)))

        # FSAL rule, first eval is last eval of previous step
        ks[:, 0] = model(t, y, **kwargs)
        ks[:, 1] = model(t + hs[0], y + h * ks[:, 0] * self.betas[0], **kwargs)
        ks[:, 2] = model(t + hs[1], y + h * np.sum(ks[:, :2] * self.betas[1], axis=1), **kwargs)
        ks[:, 3] = model(t + hs[2], y + h * np.sum(ks[:, :3] * self.betas[2], axis=1), **kwargs)
        ks[:, 4] = model(t + hs[3], y + h * np.sum(ks[:, :4] * self.betas[3], axis=1), **kwargs)
        ks[:, 5] = model(t + hs[4], y + h * np.sum(ks[:, :5] * self.betas[4], axis=1), **kwargs)

        # 5th order solution, returned in 6 evaluations
        # step size estimation does not happen here
        y_new = y + h * np.sum(ks * self.gammas, axis=1)

        new_state = {**{model.indep_name: t + h},
                     **dict(zip(model.variable_names, y_new))}

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

    @staticmethod
    def get_data_from_state_dict(model: ODEModel,
                                 state: Dict[Text, float]):

        t = state.pop(model.indep_name)

        # at this point, t is removed from the dict
        # and only the state is left
        y = np.array(list(state.values()))

        return t, y

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:
        pass


class ImplicitEulerMethod(StepFunction):
    """
    Implicit Euler Method for ODE solving.
    """
    def __init__(self):
        super(ImplicitEulerMethod, self).__init__()
        self.order = 1

    def forward(self,
                model: ODEModel,
                state: Dict[Text, float],
                h: float,
                **kwargs) -> Dict[Text, float]:
        pass
