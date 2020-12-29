# dynamic (variable step size) integration variables
MAX_STEPS = 10000
INITIAL_H = 0.01

# standard function signatures for normal ODEs
# and Hamiltonian systems
standard_rhs = ["t", "y"]
hamiltonian_rhs = ["t", "q", "p"]

# step size control builtin metric names
iteration = "iteration"
step_size = "h"
accepted = "accepted"
rejected = "rejected"
