import numpy as np

from ode_explorer.stepfunctions import *
from ode_explorer.stepfunctions import ExplicitRungeKuttaMethod
from ode_explorer.models import ODEModel
from ode_explorer.integrators import Integrator
from ode_explorer.metrics import DistanceToSolution
from math import sqrt

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

    integrator = Integrator()

    c = sqrt(3) / 6

    alphas = np.array([0.5-c, 0.5+c])
    betas = np.array([[0.25, 0.25-c],
                      [0.25+c, 0.25]])
    gammas = np.array([0.5, 0.5])

    step2 = ImplicitRungeKuttaMethod(alphas=alphas, betas=betas, gammas=gammas)

    initial_state = (t_0, y_0)

    integrator.integrate_const(model=model,
                               step_func=RungeKutta4(),
                               initial_state=initial_state,
                               h=0.001,
                               max_steps=10000,
                               verbosity=1,
                               progress_bar=True,
                               metrics=[DistanceToSolution(solution=sol, name="l2_distance")])

    metrics = integrator.return_metrics(run_id="latest")

    print(metrics.describe())

    integrator.integrate_const(model=model,
                               step_func=step2,
                               initial_state=initial_state,
                               h=0.001,
                               max_steps=10000,
                               verbosity=1,
                               progress_bar=True,
                               metrics=[DistanceToSolution(solution=sol, name="l2_distance")])

    metrics = integrator.return_metrics(run_id="latest")

    print(metrics.describe())


if __name__ == "__main__":
    main()
