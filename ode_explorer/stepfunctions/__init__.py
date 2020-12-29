from ode_explorer.stepfunctions.stepfunctions import (
    EulerMethod,
    HeunMethod,
    RungeKutta4,
    DOPRI45,
    ImplicitEulerMethod,
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