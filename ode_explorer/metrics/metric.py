from typing import Dict, Text, Any, Callable, Union

import jax.numpy as jnp

from ode_explorer.models import BaseModel
from ode_explorer.types import State


class Metric:
    """
    Base metric interface. Subclass this to define your own metrics,
    to be computed during ODE integration.
    """

    def __init__(self, name: Text = None):
        """
        Base metric constructor.

        Args:
            name: Optional name identifier. This is the name that will be displayed
             in metrics data frames obtained from integration runs.
        """
        self.__name__ = name or self.__class__.__name__

    def __call__(self,
                 i: int,
                 state: State,
                 new_state: State,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> Any:
        raise NotImplementedError


class DistanceToSolution(Metric):
    """
    Tracks distance of an ODE state vector to the state vector of a known
    solution. This is useful to test whether an integrator performs as
    expected.
    """

    def __init__(self,
                 solution: Callable,
                 norm: Union[Text, int] = None,
                 name: Text = None):
        """
        Solution distance metric constructor.

        Args:
            solution: Callable, of signature t -> y(t) giving the ODE solution at time t.
            norm: Norm identifier for use in jnp.linalg.norm.
            name: Optional name identifier.
        """
        super(DistanceToSolution, self).__init__(name=name)

        self.solution = solution
        self.norm = norm or None

    def __call__(self,
                 i: int,
                 state: State,
                 new_state: State,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> Any:
        """
        Solution distance call operator overload.

        Args:
            i: Current iteration number.
            state: Previous ODE model state.
            new_state: New calculated ODE model state.
            model: ODE model that is being integrated.
            local_vars: Handle for locals() dict object.

        Returns:
            A scalar, the norm difference between the calculated state and the theoretical solution.

        """
        t, y = new_state

        y_pred = self.solution(t)

        return jnp.linalg.norm(y - y_pred, ord=self.norm)
