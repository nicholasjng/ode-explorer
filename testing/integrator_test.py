import numpy as np
from typing import Union

from ode_explorer.stepfunctions import *
from ode_explorer.model import ODEModel
from ode_explorer.integrator import Integrator


def ode_func(t: float, y: Union[float, np.ndarray], lamb: float):
    return - lamb * y


def main(argv):
    t_0 = 0.0
    y_0 = 1.0  # np.ones(100)
    lamb = 0.5

    model = ODEModel(ode_fn=ode_func, fn_args={"lamb": lamb})

    integrator = Integrator()

    initial_state = {"t": t_0, "y": y_0}

    integrator.integrate_const(model=model,
                               step_func=RungeKutta4(),
                               initial_state=initial_state,
                               h=0.001,
                               num_steps=10000,
                               verbosity=1)


if __name__ == "__main__":
    # app.run(main)
    main(None)
