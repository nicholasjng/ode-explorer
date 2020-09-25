import numpy as np
import absl

from absl import logging
from absl import app

from ode_explorer.stepfunctions import ImplicitEulerMethod, RungeKutta4
from ode_explorer.model import ODEModel
from ode_explorer.integrator import Integrator


def ode_func(t: float, y: float, _lambda: float):
    return -_lambda * y


def main(argv):
    t_0 = 0.0
    y_0 = np.ones(100)
    l = 0.5

    model = ODEModel(ode_fn=ode_func, fn_args={"_lambda": l})

    integrator = Integrator(step_func=RungeKutta4())

    initial_state = {"t": t_0, "y": y_0}

    integrator.integrate_const(model=model, initial_state=initial_state,
                               h=0.001, num_steps=10000, verbosity=1)

    integrator = Integrator(step_func=RungeKutta4(cache_ks=True))

    integrator.integrate_const(model=model, initial_state=initial_state,
                               h=0.001, num_steps=10000, verbosity=1)

if __name__ == "__main__":
    app.run(main)
    #main(None)