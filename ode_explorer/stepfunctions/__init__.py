from ode_explorer.stepfunctions.stepfunctions import (
    ForwardEulerMethod,
    HeunMethod,
    RungeKutta4,
    DOPRI45,
    BackwardEulerMethod,
    AdamsBashforth2,
    BDF2
)

from ode_explorer.stepfunctions.templates import (
    SingleStepMethod,
    MultiStepMethod,
    ExplicitRungeKuttaMethod,
    ImplicitRungeKuttaMethod,
    ExplicitMultiStepMethod,
    ImplicitMultiStepMethod
)

from typing import Union

StepFunction = Union[SingleStepMethod, MultiStepMethod]