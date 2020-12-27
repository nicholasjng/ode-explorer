from ode_explorer.stepfunctions.stepfunctions import (
    EulerMethod,
    EulerCython,
    HeunMethod,
    RungeKutta4,
    DOPRI5,
    DOPRI45,
    ImplicitEulerMethod,
    AdamsBashforth2
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