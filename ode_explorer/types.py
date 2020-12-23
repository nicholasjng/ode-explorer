from typing import Tuple, Union

import numpy as np

StateVariable = Union[float, np.ndarray]

ModelState = Tuple[StateVariable, ...]
