from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import types
from typing import Text, Callable

__all__ = ["import_func_from_source", "import_func_from_module"]


def import_func_from_source(source_path: Text, fn_name: Text) -> Callable:
    """Imports a function from a module provided as source file."""

    try:
        loader = importlib.machinery.SourceFileLoader(
            fullname='user_module',
            path=source_path,
        )
        user_module = types.ModuleType(loader.name)
        loader.exec_module(user_module)
        return getattr(user_module, fn_name)

    except IOError:
        raise ImportError('{} in {} not found in '
                          'import_func_from_source()'.format(fn_name,
                                                             source_path))


def import_func_from_module(module_path: Text, fn_name: Text) -> Callable:
    """
    Imports a function from a module provided as source file or module path.
    """
    user_module = importlib.import_module(module_path)
    return getattr(user_module, fn_name)
