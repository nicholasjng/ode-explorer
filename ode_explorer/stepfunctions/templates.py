import logging

import numpy as np
from scipy.optimize import root

from ode_explorer.models import BaseModel, ODEModel
from ode_explorer.types import StateVariable, ModelState
from ode_explorer.utils.helpers import is_scalar

logger = logging.getLogger(__name__)

__all__ = ["SingleStepMethod",
           "MultiStepMethod",
           "ExplicitRungeKuttaMethod",
           "ImplicitRungeKuttaMethod",
           "ExplicitMultiStepMethod",
           "ImplicitMultiStepMethod"]


class SingleStepMethod:
    """
    Base class for all single step functions for ODE solving. Override this class and its methods
    to make your own custom single-step functions.
    """

    def __init__(self, order: int = 0):
        """
        Base SingleStepMethod constructor.

        Args:
            order: Order of the method.
        """
        self.order = order
        self.model_dim = 0
        self.num_stages = 0

    def _adjust_dims(self, y: StateVariable):
        scalar_ode = is_scalar(y)

        if scalar_ode:
            model_dim = 1
            shape = (self.num_stages,)
        else:
            model_dim = len(y)
            shape = (self.num_stages, model_dim)

        self.model_dim = model_dim
        self.k = np.zeros(shape=shape)

    def _get_shape(self, y: StateVariable):
        return (self.num_stages,) if is_scalar(y) else (self.num_stages, len(y))

    @staticmethod
    def get_data_from_state(state: ModelState):
        """
        Custom member function for getting the raw numpy-compatible data from a ModelState object.
        Override this if you intend to use a custom state type such as a NamedTuple.

        Args:
            state: State object holding the numpy-compatible data.

        Returns:
            Raw numpy-compatible state data for use in the forward member function.
        """
        return state

    @staticmethod
    def make_new_state(t: StateVariable, y: StateVariable) -> ModelState:
        """
        Custom function for constructing a new state from numpy data.
        Override this if you intend to use a custom state type such as a NamedTuple.

        Args:
            t: Time variable at the new state.
            y: Spatial variable at the new state.

        Returns:
            A new state object holding the raw data.
        """
        return t, y

    def reset(self):
        """
        Unused reset method for compatibility.
        """
        pass

    def forward(self,
                model: BaseModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state using a single-step method.
        Override this to define your own single-step functions.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """
        raise NotImplementedError


class MultiStepMethod:
    """
    Base class for explicit multi-step methods for ODE solving. Override this class and its methods
    to make your own custom multi-step functions.

    Multi-step methods are defined by two sets of coefficients, commonly denoted a and b in numerical
    literature. For more information and sample methods, see
    https://en.wikipedia.org/wiki/Linear_multistep_method.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 a_coeffs: np.ndarray,
                 b_coeffs: np.ndarray,
                 order: int = 0,
                 reverse: bool = True):
        """
        Base MultiStepMethod constructor.

        Args:
            startup: SingleStepMethod used to compute the startup data.
            a_coeffs: Array of a-coefficients of the method.
            b_coeffs: Array of b-coefficients of the method.
            order: Order of the method.
            reverse: Whether to reverse the coefficient arrays. Set this to True if you are copying
             coefficient sets e.g. from Wikipedia, as they usually count the states in reverse.
        """
        self.order = order

        # startup calculation variables, only for multi-step methods
        self.ready = False
        self._cache_idx = 1
        self.startup = startup

        if reverse:
            self.a_coeffs = np.flip(a_coeffs)
            self.b_coeffs = np.flip(b_coeffs)
        else:
            self.a_coeffs = a_coeffs
            self.b_coeffs = b_coeffs

        # TODO: This is not good
        self.num_previous = max(len(b_coeffs), len(a_coeffs))

        # side cache for additional steps
        self.f_cache = np.zeros(self.num_previous)
        self.t_cache = np.zeros(self.num_previous)
        self.y_cache = np.zeros(self.num_previous)

    @staticmethod
    def get_data_from_state(state: ModelState):
        """
        Custom member function for getting the raw numpy-compatible data from a ModelState object.
        Override this if you intend to use a custom state type such as a NamedTuple.

        Args:
            state: State object holding the numpy-compatible data.

        Returns:
            Raw numpy-compatible state data for use in the forward member function.
        """
        return state

    @staticmethod
    def make_new_state(t: StateVariable, y: StateVariable) -> ModelState:
        """
        Custom function for constructing a new state from numpy data.
        Override this if you intend to use a custom state type such as a NamedTuple.

        Args:
            t: Time variable at the new state.
            y: Spatial variable at the new state.

        Returns:
            A new state object holding the raw data.
        """
        return t, y

    def _adjust_dims(self, y: StateVariable):
        if is_scalar(y):
            model_dim = 1
            shape = (self.num_previous,)

        else:
            model_dim = len(y)
            shape = (self.num_previous, model_dim)

        self.model_dim = model_dim
        self.f_cache = np.zeros(shape=shape)
        self.y_cache = np.zeros(shape=shape)

    def _get_shape(self, y: StateVariable):
        return (self.num_previous,) if is_scalar(y) else (self.num_previous, len(y))

    def _increment_cache_idx(self):
        self._cache_idx += 1

    def reset(self):
        """
        Resets the step function so that next time the multi-step method is called,
        new startup values will be calculated with the saved startup step
        function. Useful if the multi-step method will be reused in
        multiple non-consecutive runs with different model dimensions.
        """
        self.ready = False
        self._cache_idx = 1

    def _perform_startup_calculation(self,
                                     model: ODEModel,
                                     state: ModelState,
                                     h: float,
                                     **kwargs):
        """
        Performs the startup calculation using the startup step function supplied at construction.
        Some startup values need to be computed in the first step in order to obtain enough starting
        values for the multi-step procedure. The number of startup steps is dependent on the number of
        stages in the method.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.
        """

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.y_cache.shape:
            self._adjust_dims(y)

        # fill function evaluation cache
        self.t_cache[0], self.y_cache[0], self.f_cache[0] = t, y, model(t, y)

        for i in range(1, self.num_previous):
            startup_state = self.startup.forward(model=model,
                                                 state=state,
                                                 h=h,
                                                 **kwargs)

            self.t_cache[i], self.y_cache[i] = startup_state
            self.f_cache[i] = model(self.t_cache[i], self.y_cache[i])
            state = startup_state

        self.ready = True
        self._increment_cache_idx()

    def _get_cached_state(self):
        idx = self._cache_idx
        self._increment_cache_idx()
        return self.make_new_state(t=self.t_cache[idx], y=self.y_cache[idx])

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state with a multi-step method.
        Override this to define your own multi-step functions.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """
        raise NotImplementedError


class ExplicitRungeKuttaMethod(SingleStepMethod):
    """
    Base class template for explicit Runge-Kutta (RK) methods.

    A Runge-Kutta method is a generalized s-stage algorithm for advancing an ODE in time.
    It is defined by three sets of coefficients commonly called a Butcher tableau.
    An explicit Runge-Kutta method is characterized by a strictly lower-diagonal b-coefficient matrix.

    For more information on Runge-Kutta methods and the Butcher tableau, see
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods.
    """
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 order: int = 0):
        """
        Explicit Runge-Kutta method constructor.

        Args:
            alphas: Alpha- or a-array in the Butcher tableau (commonly the left column).
            betas: Beta- or b-matrix in the Butcher tableau (commonly in the upper right).
            gammas: Gamma- or c-array in the Butcher tableau (commonly the bottom row).
            order: Order of the resulting explicit RK method.
        """

        super(ExplicitRungeKuttaMethod, self).__init__(order=order)

        self._validate_butcher_tableau(alphas=alphas, betas=betas, gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.num_stages = len(self.alphas)
        self.k = np.zeros(betas.shape[0])

    @staticmethod
    def _validate_butcher_tableau(alphas: np.ndarray,
                                  betas: np.ndarray,
                                  gammas: np.ndarray) -> None:
        _error_msg = []
        if len(alphas) != len(gammas):
            _error_msg.append("Alpha and gamma vectors are not the same length")

        if betas.shape[0] != betas.shape[1]:
            _error_msg.append("Betas must be a quadratic matrix with the same "
                              "dimension as the alphas/gammas arrays")

        # for an explicit method, betas must be lower triangular
        if not np.allclose(betas, np.tril(betas, k=-1)):
            _error_msg.append("The beta matrix has to be lower triangular for "
                              "an explicit Runge-Kutta method, i.e. "
                              "b_ij = 0 for i <= j")

        if _error_msg:
            raise ValueError("An error occurred while validating the input "
                             "Butcher tableau. More information: "
                             "{}.".format(",".join(_error_msg)))

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state with a
        multi-stage explicit Runge-Kutta method.

        This function is templated and not meant to be directly overridden. If you want more
        control over your step function, consider implementing an explicit RK method by subclassing the
        ``SingleStepMethod`` class.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.k.shape:
            self._adjust_dims(y)

        self.k[0] = model(t, y)

        for i in range(1, self.num_stages):
            # first row of betas is a zero row because it is an explicit RK
            self.k[i] = model(t + h * self.alphas[i], y + h * np.dot(self.betas[i], self.k))

        y_new = y + h * np.dot(self.gammas, self.k)

        return self.make_new_state(t=t + h, y=y_new)


class ImplicitRungeKuttaMethod(SingleStepMethod):
    """
    Base class template for implicit Runge-Kutta (RK) methods.

    A Runge-Kutta method is a generalized s-stage algorithm for advancing an ODE in time.
    It is defined by three sets of coefficients commonly called a Butcher tableau.

    An implicit Runge-Kutta method incurs generally much more computational effort than an explicit one,
    as a non-linear system of equations needs to be solved in each step. However, implicit methods
    have better properties when used on stiff equations, and can achieve very high order with a
    comparably low number of stages s.

    For more information on implicit Runge-Kutta methods and the Butcher tableau, see
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods.
    """
    def __init__(self,
                 alphas: np.ndarray,
                 betas: np.ndarray,
                 gammas: np.ndarray,
                 order: int = 0,
                 **kwargs):
        """
        Implicit Runge-Kutta method constructor.

        Args:
            alphas: Alpha- or a-array in the Butcher tableau (commonly the left column).
            betas: Beta- or b-matrix in the Butcher tableau (commonly in the upper right).
            gammas: Gamma- or c-array in the Butcher tableau (commonly the bottom row).
            order: Order of the resulting implicit RK method.
            **kwargs: Additional keyword arguments used in the call to scipy.optimize.root.
        """

        super(ImplicitRungeKuttaMethod, self).__init__(order=order)

        self.validate_butcher_tableau(alphas=alphas, betas=betas, gammas=gammas)

        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.num_stages = len(self.alphas)
        self.k = np.zeros(betas.shape[0])

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

        self._array_ops = {"scalar": np.array,
                           "ndim": np.concatenate}

    @staticmethod
    def validate_butcher_tableau(alphas: np.ndarray,
                                 betas: np.ndarray,
                                 gammas: np.ndarray) -> None:
        _error_msg = []
        if len(alphas) != len(gammas):
            _error_msg.append("Alpha and gamma vectors are "
                              "not the same length")

        if betas.shape[0] != betas.shape[1]:
            _error_msg.append("Betas must be a quadratic matrix with the same "
                              "dimension as the alphas/gammas arrays")

        if betas.shape[0] == 1:
            _error_msg.append("You have supplied a single-stage implicit RK method. Please use the "
                              "builtin BackwardEulerMethod class instead.")

        if _error_msg:
            raise ValueError("An error occurred while validating the input "
                             "Butcher tableau. More information: "
                             "{}.".format(",".join(_error_msg)))

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state with a
        multi-stage implicit Runge-Kutta method.

        This function is templated and not meant to be directly overridden. If you want more
        control over your step function, consider implementing an implicit RK method by subclassing the
        ``SingleStepMethod`` class instead.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """

        t, y = self.get_data_from_state(state=state)

        if self._get_shape(y) != self.k.shape:
            self._adjust_dims(y)

        initial_shape = self.k.shape
        shape_prod = np.prod(initial_shape)

        op_type = "scalar" if is_scalar(y) else "ndim"

        def F(x: np.ndarray) -> np.ndarray:
            # kwargs are not allowed in scipy.optimize, so pass tuple instead
            model_stack = self._array_ops.get(op_type)(
                [model(t + h * self.alphas[i], y + h * np.dot(self.betas[i], x.reshape(initial_shape)))
                 for i in range(self.num_stages)])

            return model_stack - x

        # sort the kwargs before putting them into the tuple passed to root
        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        root_res = root(F, x0=self.k.reshape((shape_prod,)), args=args, **self.solver_kwargs)

        y_new = y + h * np.dot(self.gammas, root_res.x.reshape(initial_shape))

        return self.make_new_state(t=t + h, y=y_new)


class ExplicitMultiStepMethod(MultiStepMethod):
    """
    Base class for explicit multi-step methods for ODE solving.

    In contrast to a single-step method, a linear multi-step method uses multiple past values
    to compute a new ODE state.

    An explicit multi-step method is characterized by a trivial array of a-coefficients. The most
    prominent example of an explicit multi-step method is the class of Adams-Bashforth methods.

    For more information on linear multi-step methods and sample coefficient sets, see
    https://en.wikipedia.org/wiki/Linear_multistep_method.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 a_coeffs: np.ndarray,
                 b_coeffs: np.ndarray,
                 order: int = 0,
                 reverse: bool = True):
        """
        Base ExplicitMultiStepMethod constructor.

        Args:
            startup: Single-step method used to compute the startup values.
            a_coeffs: Array of a-coefficients. Unused for explicit multi-step methods.
            b_coeffs: Array of b-coefficients.
            order: Order of the resulting explicit multi-step method.
            reverse: Whether to reverse the coefficient arrays. Set this to True if you are copying
             coefficient sets e.g. from Wikipedia, as they usually count the states in reverse.
        """

        super(ExplicitMultiStepMethod, self).__init__(startup=startup,
                                                      a_coeffs=a_coeffs,
                                                      b_coeffs=b_coeffs,
                                                      order=order,
                                                      reverse=reverse)

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state with an explicit
        multi-step method.

        This function is templated and not meant to be directly overridden. If you want more
        control over your step function, consider implementing an explicit multi-step method
        by subclassing the ``SingleStepMethod`` class instead.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """

        if not self.ready:
            # startup calculation to the multi-step method,
            # fills the y-, t- and f-caches
            self._perform_startup_calculation(model=model,
                                              state=state,
                                              h=h,
                                              **kwargs)

            return self.make_new_state(self.t_cache[1], self.y_cache[1])

        t, y = self.get_data_from_state(state=state)

        if self._cache_idx < self.num_previous:
            return self._get_cached_state()

        y_new = y + h * np.dot(self.b_coeffs, self.f_cache)

        self.f_cache = np.roll(self.f_cache, shift=-1, axis=0)
        self.f_cache[-1] = model(t + h, y_new)

        return self.make_new_state(t=t + h, y=y_new)


class ImplicitMultiStepMethod(MultiStepMethod):
    """
    Base class for explicit multi-step methods for ODE solving.

    In contrast to a single-step method, a linear multi-step method uses multiple past values
    to compute a new ODE state.

    An explicit multi-step method is characterized by a trivial array of a-coefficients. The most
    prominent examples of implicit multi-step methods are the Adams-Moulton methods and the
    backward differentiation formulas (BDFs).

    For more information on linear multi-step methods and sample coefficient sets, see
    https://en.wikipedia.org/wiki/Linear_multistep_method.
    """

    def __init__(self,
                 startup: SingleStepMethod,
                 a_coeffs: np.ndarray,
                 b_coeffs: np.ndarray,
                 order: int = 0,
                 reverse: bool = True,
                 **kwargs):
        """
        ImplicitMultiStepMethod constructor.

        Args:
            startup: Single-step method used to compute the startup values.
            a_coeffs: Array of a-coefficients. Unused for explicit multi-step methods.
            b_coeffs: Array of b-coefficients.
            order: Order of the resulting explicit multi-step method.
            reverse: Whether to reverse the coefficient arrays. Set this to True if you are copying
             coefficient sets e.g. from Wikipedia, as they usually count the states in reverse.
            **kwargs: Additional keyword arguments used in the call to scipy.optimize.root.
        """

        super(ImplicitMultiStepMethod, self).__init__(startup=startup,
                                                      a_coeffs=a_coeffs,
                                                      b_coeffs=b_coeffs,
                                                      order=order,
                                                      reverse=reverse)

        # scipy.optimize.root options
        self.solver_kwargs = kwargs

    def forward(self,
                model: ODEModel,
                state: ModelState,
                h: float,
                **kwargs) -> ModelState:
        """
        Main method to advance an ODE in time by computing a new state with an implicit
        multi-step method.

        This function is templated and not meant to be directly overridden. If you want more
        control over your step function, consider implementing an implicit multi-step method by
        subclassing the ``SingleStepMethod`` class instead.

        Args:
            model: ODEModel object implementing the ODE model.
            state: Input state.
            h: Step size to use in the step function.
            **kwargs: Additional keyword arguments, unused for now.

        Returns:
            A new state containing the ODE model data at time t+h.
        """

        if not self.ready:
            # startup calculation to the multi-step method,
            # fills the state and f-caches
            self._perform_startup_calculation(model=model,
                                              state=state,
                                              h=h,
                                              **kwargs)

            # first cached value
            return self.make_new_state(self.t_cache[1], self.y_cache[1])

        b = self.b_coeffs[-1]

        t, y = self.get_data_from_state(state=state)

        if self._cache_idx < self.num_previous:
            return self._get_cached_state()

        def F(x: StateVariable) -> StateVariable:
            return x + np.dot(self.a_coeffs, self.y_cache) - h * b * model(t + h, x)

        if kwargs:
            args = tuple(kwargs[arg] for arg in model.fn_args.keys())
        else:
            args = ()

        # TODO: Retry here in case of convergence failure?
        root_res = root(F, x0=y, args=args, **self.solver_kwargs)

        y_new = root_res.x

        self.y_cache = np.roll(self.y_cache, shift=-1, axis=0)
        self.y_cache[-1] = y_new

        return self.make_new_state(t=t + h, y=y_new)
