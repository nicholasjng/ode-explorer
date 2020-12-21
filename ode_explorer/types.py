from typing import Tuple, Text, Dict, Union, Callable, Any

import numpy as np

StateVariable = Union[float, np.ndarray]

AdditionalArgs = Any

ODEFunction = Callable[[StateVariable, StateVariable, AdditionalArgs], StateVariable]

ModelState = Tuple[StateVariable, ...]
