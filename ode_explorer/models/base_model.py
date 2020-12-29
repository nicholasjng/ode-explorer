
class BaseModel:
    """
    Base model class. Override this to define your own ODE model classes.
    """
    def make_state(self, *args, **kwargs):
        """
        Constructs a state object from raw input floats and numpy arrays.

        Returns:
            A state object holding the input data.

        """
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        BaseModel call operator. Overload this to use your model with builtin step functions.

        Returns:
            A state vector corresponding to the right hand side of y' = f(t,y).

        """
        raise NotImplementedError
