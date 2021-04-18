import jax.numpy as jnp

from ode_explorer.stepfunctions import *
from ode_explorer.models import HamiltonianSystem
from ode_explorer.integrators import Integrator


def free_particle_hamiltonian(t: float, q: jnp.ndarray, p: jnp.ndarray, m=1.0) -> float:
    kinetic_energy = jnp.dot(p, p) / (2*m)
    potential = 0.0
    return kinetic_energy + potential


def q_deriv(t, q, m=1.0):
    return jnp.zeros_like(q, q.dtype)


def p_deriv(t, p, m=1.0):
    return p / m


def main():
    t_0 = 0.0
    q_0 = jnp.zeros(2)
    p_0 = jnp.ones(2) / jnp.sqrt(2)

    mass = 2.0

    model = HamiltonianSystem(hamiltonian=free_particle_hamiltonian,
                              p_derivative=p_deriv,
                              q_derivative=q_deriv,
                              h_args={"m": mass})

    integrator = Integrator()

    initial_state = (t_0, q_0, p_0)

    integrator.integrate_const(model=model,
                               step_func=EulerA(),
                               initial_state=initial_state,
                               h=0.001,
                               max_steps=100,
                               verbosity=1,
                               progress_bar=True,
                               output_dir="hamiltonian_test")


if __name__ == "__main__":
    main()
