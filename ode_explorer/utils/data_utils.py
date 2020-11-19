import numpy as np
import pandas as pd
import os
from ode_explorer.model import ODEModel
from ode_explorer.constants import DataFormatKeys
from ode_explorer.utils.helpers import is_scalar
from typing import List, Dict, Text, Any, Union

__all__ = ["convert_to_zipped", "convert_from_zipped", "convert_state_dict",
           "infer_dict_format", "write_to_file", "make_log_dir"]


def make_log_dir():
    pass


def convert_to_zipped(state_dict: Dict[Text, Any], model: ODEModel):
    t = state_dict[model.indep_name]
    y = state_dict[model.variable_names[0]]

    # small check for a default value in the scalar case (n = 1)
    if is_scalar(y):
        return {model.indep_name: t, model.dim_names[0]: y}
    return {model.indep_name: t, **dict(zip(model.dim_names, y))}


def convert_from_zipped(state_dict: Dict[Text, Any], model: ODEModel):
    t = state_dict[model.indep_name]
    y = np.array([state_dict[key] for key in model.dim_names])

    if len(model.dim_names) == 1:
        return {model.indep_name: t, model.variable_names[0]: y[0]}
    return {model.indep_name: t, model.variable_names[0]: y}


def convert_state_dict(state_dict: Dict[Text, Any], model: ODEModel,
                       input_format: Text = None,
                       output_format: Text = DataFormatKeys.ZIPPED):
    # infer dict mode by presence of variable name
    if not input_format:
        input_format = infer_dict_format(state_dict=state_dict, model=model)

    if input_format == output_format:
        return state_dict

    if output_format == DataFormatKeys.ZIPPED:
        return convert_to_zipped(state_dict=state_dict, model=model)
    else:
        return convert_from_zipped(state_dict=state_dict, model=model)


def infer_dict_format(state_dict: Dict[Text, Union[float, np.ndarray]],
                      model: ODEModel):

    if all(var in state_dict for var in model.variable_names):
        # this is the case of a scalar ODE, where ZIPPED and VARIABLES are
        # the same
        if not any(isinstance(v, np.ndarray) for v in state_dict.values()):
            return DataFormatKeys.ZIPPED
        return DataFormatKeys.VARIABLES
    elif all(dim in state_dict for dim in model.dim_names):
        return DataFormatKeys.ZIPPED
    else:
        raise ValueError("Error: Unrecognizable state dict format. Something "
                         "went wrong.")


def write_to_file(result_data: List[Dict[Text, float]],
                  model: ODEModel,
                  out_dir: Text,
                  outfile_name: Text,
                  **kwargs) -> None:
    """
    Small wrapper to write ODE simulation results to a .csv file.
    :param model: ODE model.
    :param result_data: Result data to be written to file.
    :param out_dir: Log output directory
    :param outfile_name: Log output file name.
    :param kwargs: Additional keyword arguments, passed to pandas.to_csv.
    :return: None.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_ext = ".csv"

    for i, res in enumerate(result_data):
        result_data[i] = convert_to_zipped(res, model)

    # convert result_list to data frame
    # fast construction from list with schema dict
    result_df = pd.DataFrame(data=result_data)

    out_file = os.path.join(out_dir, outfile_name)

    result_df.to_csv(out_file + file_ext, **kwargs)
