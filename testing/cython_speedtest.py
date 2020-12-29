import numpy as np

from ode_explorer.stepfunctions import *
from ode_explorer.models import ODEModel
from ode_explorer.integrators import Integrator


def ode_func(t: float, y: Union[float, np.ndarray], lamb: float = 0.5):
    return - lamb * y


def main():
    t_0 = 0.0
    # y_0 = 1.0
    y_0 = np.ones(100)
    lamb = 0.5

    model = ODEModel(ode_fn=ode_func, fn_args={"lamb": lamb})

    integrator = Integrator()

    initial_state = (t_0, y_0)

    integrator.integrate_const(model=model,
                               step_func=ForwardEulerMethod(),
                               initial_state=initial_state,
                               h=0.001,
                               max_steps=10000,
                               verbosity=1,
                               progress_bar=True)


if __name__ == "__main__":
    main()
