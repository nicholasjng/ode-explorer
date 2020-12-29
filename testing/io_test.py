import numpy as np
import logging

from ode_explorer.stepfunctions import *
from ode_explorer.models import ODEModel
from ode_explorer.integrators import Integrator
from ode_explorer.metrics import DistanceToSolution

y_0 = 1.0
lamb = 0.5


def ode_func(t: float, y: Union[float, np.ndarray], lamb: float = 0.5):
    return - lamb * y


def sol(t):
    return y_0 * np.exp(-lamb * t)


def main():
    t_0 = 0.0

    model = ODEModel(ode_fn=ode_func, fn_args={"lamb": lamb})

    step_func = ForwardEulerMethod()

    initial_state = (t_0, y_0)

    integrator = Integrator()

    integrator.integrate_const(model=model,
                               step_func=step_func,
                               initial_state=initial_state,
                               h=0.01,
                               max_steps=5,
                               verbosity=logging.INFO,
                               progress_bar=True,
                               metrics=[DistanceToSolution(solution=sol, name="l2_distance")],
                               output_dir="my_run123")


if __name__ == "__main__":
    main()
