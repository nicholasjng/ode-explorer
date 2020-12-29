import numpy as np

from ode_explorer.integrators import Integrator
from ode_explorer.metrics import DistanceToSolution
from ode_explorer.models import ODEModel
from ode_explorer.stepfunctions import *
from ode_explorer.stepfunctions import ExplicitRungeKuttaMethod

y_0_scalar = 1.0
y_0_vec = np.ones(10)
lamb = 0.5


def ode_func(t: float, y: Union[float, np.ndarray], lamb: float = 0.5):
    return - lamb * y


def sol_scalar(t):
    return y_0_scalar * np.exp(-lamb * t)


def sol_vec(t):
    return y_0_vec * np.exp(-lamb * t)


def main():
    t_0 = 0.0

    model = ODEModel(ode_fn=ode_func, fn_args={"lamb": lamb})

    integrator = Integrator()

    step_funcs = []

    rk4_alphas = np.array([0.0, 0.5, 0.5, 1.0])
    rk4_betas = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]])
    rk4_gammas = np.array([1.0, 2.0, 2.0, 1.0]) / 6

    templated_rk4 = ExplicitRungeKuttaMethod(alphas=rk4_alphas, betas=rk4_betas, gammas=rk4_gammas)

    step_funcs.append(templated_rk4)

    c = np.sqrt(3) / 6

    i_alphas = np.array([0.5 - c, 0.5 + c])
    i_betas = np.array([[0.25, 0.25 - c],
                        [0.25 + c, 0.25]])
    i_gammas = np.array([0.5, 0.5])

    gauss_lobatto2 = ImplicitRungeKuttaMethod(alphas=i_alphas, betas=i_betas, gammas=i_gammas)

    step_funcs.append(gauss_lobatto2)

    for initial in [(y_0_scalar, sol_scalar), (y_0_vec, sol_vec)]:
        initial_y, sol = initial
        initial_state = (t_0, initial_y)

        for step_func in step_funcs:
            step_func.reset()
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


if __name__ == "__main__":
    main()
