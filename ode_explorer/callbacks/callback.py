import logging
from typing import Dict, Text, Any

from ode_explorer.models import BaseModel
from ode_explorer.types import State

logger = logging.getLogger(__name__)


class Callback:
    """
    Base callback interface. Callbacks can be used to change the control flow of an
    integration run for an ODE model.
    """
    def __init__(self, name: Text = None):
        """
        Base callback constructor.

        Args:
            name: Optional string identifier.
        """
        self.__name__ = name or self.__class__.__name__

    def __call__(self,
                 i: int,
                 state: State,
                 new_state: State,
                 model: BaseModel,
                 local_vars: Dict[Text, Any]) -> None:
        """
        Callback class call operator. Overload this with your custom logic to use in
        ODE integration runs.

        Args:
            i: Current iteration number.
            state: Previous ODE state.
            new_state: New ODE state calculated by the used step function.
            model: ODE model that is used in the integration run.
            local_vars: Handle for the locals() dict passed to the Callbacks.
        """
        raise NotImplementedError
