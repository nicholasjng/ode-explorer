import os
from typing import List, Dict, Text, Any

import pandas as pd

from ode_explorer.constants import ModelMetadataKeys
from ode_explorer.types import ModelState
from ode_explorer.utils.helpers import is_scalar

__all__ = ["convert_to_dict", "write_to_file"]


def convert_to_dict(state: ModelState, model_metadata: Dict[Text, Any]):
    output_dict = dict()

    variable_names = model_metadata[ModelMetadataKeys.VARIABLE_NAMES]
    dim_names = model_metadata[ModelMetadataKeys.DIM_NAMES]

    idx = 0
    for i, name in enumerate(variable_names):
        v = state[i]

        if is_scalar(v):
            k = dim_names[idx]
            output_dict.update({k: v})
            idx += 1
        else:
            k = dim_names[idx:idx + len(v)]
            output_dict.update(dict(zip(k, v)))
            idx += len(v)

    return output_dict


def write_to_file(result_data: List[ModelState],
                  model_metadata: Dict[Text, Any],
                  out_dir: Text,
                  outfile_name: Text,
                  **kwargs) -> None:
    """
    Small wrapper to write ODE simulation results to a .csv file.
    :param model_metadata: ODE model metadata.
    :param result_data: Result data to be written to file.
    :param out_dir: Log output directory
    :param outfile_name: Log output file name.
    :param kwargs: Additional keyword arguments, passed to pandas.to_csv.
    :return: None.
    """
    data = []

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_ext = ".csv"

    for i, res in enumerate(result_data):
        data[i] = convert_to_dict(res, model_metadata=model_metadata)

    # convert result_list to data frame
    # fast construction from list with schema dict
    result_df = pd.DataFrame(data=data)

    out_file = os.path.join(out_dir, outfile_name)

    result_df.to_csv(out_file + file_ext, **kwargs)
