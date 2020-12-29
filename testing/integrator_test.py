import numpy as np

from ode_explorer.stepfunctions import *
from ode_explorer.models import ODEModel
from ode_explorer.integrators import Integrator
from ode_explorer.stepsize_control import DOPRI45Controller
from ode_explorer.metrics import DistanceToSolution

# y_0 = 1.0
y_0 = np.ones(10)
lamb = 0.5


def ode_func(t: float, y: Union[float, np.ndarray], lamb: float = 0.5):
    return - lamb * y


def sol(t):
    return y_0 * np.exp(-lamb * t)


def main():
    t_0 = 0.0

    model = ODEModel(ode_fn=ode_func, fn_args={"lamb": lamb})

    step_list = [ForwardEulerMethod(),
                 HeunMethod(),
                 RungeKutta4(),
                 BackwardEulerMethod(),
                 AdamsBashforth2(startup=ForwardEulerMethod()),
                 BDF2(startup=ForwardEulerMethod())]

    integrator = Integrator()

    initial_state = (t_0, y_0)

    for step_func in step_list:
        integrator.integrate_const(model=model,
                                   step_func=step_func,
                                   initial_state=initial_state,
                                   h=0.001,
                                   max_steps=10000,
                                   verbosity=1,
                                   progress_bar=True,
                                   metrics=[DistanceToSolution(solution=sol, name="l2_distance")])

        metrics = integrator.return_metrics(run_id="latest")

        print(metrics.describe())

    integrator.integrate_dynamically(model=model,
                                     step_func=DOPRI45(),
                                     sc=DOPRI45Controller(atol=1e-9),
                                     initial_state=initial_state,
                                     initial_h=0.01,
                                     verbosity=1,
                                     end=10.0,
                                     progress_bar=True,
                                     metrics=[DistanceToSolution(solution=sol, name="l2_distance")])

    metrics = integrator.return_metrics(run_id="latest")

    print(metrics.describe())


if __name__ == "__main__":
    main()
