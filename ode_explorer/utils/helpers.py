
__all__ = ["is_scalar"]


def is_scalar(y):
    return not hasattr(y, "__len__")