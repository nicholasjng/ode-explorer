from ode_explorer.integrators import integrator_loops as loops

loop_factory = {"constant": loops.constant_h_loop,
                "adaptive": loops.adaptive_h_loop}
