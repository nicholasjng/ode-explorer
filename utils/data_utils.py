import numpy as np
import pandas as pd
import os
import datetime
from ode_explorer.model import ODEModel
from typing import List, Dict, Text, Any


def make_log_dir():
    pass


def isscalar(y):
    return hasattr(y, "__len__")


def convert_to_zipped(state_dict: Dict[Text, Any], model: ODEModel):
    t = state_dict[model.indep_name]
    y = state_dict[model.variable_name]
    # small check for a default value in the scalar case (n = 1)
    if not hasattr(y, "__len__"):
        return {model.indep_name: t, **dict(zip(model.dim_names, [y]))}
    return {model.indep_name: t, **dict(zip(model.dim_names, y))}


def convert_from_zipped(state_dict: Dict[Text, Any], model: ODEModel):
    t = state_dict[model.indep_name]
    y = np.array([state_dict[key] for key in model.dim_names])
    if len(model.dim_names) == 1:
        return {model.indep_name: t, model.variable_name: y[0]}
    return {model.indep_name: t, model.variable_name: y}


def convert_state_dict(state_dict: Dict[Text, Any], model: ODEModel):
    # infer dict mode by presence of variable name
    if model.variable_name in state_dict.keys():
        return convert_to_zipped(state_dict=state_dict, model=model)
    else:
        return convert_from_zipped(state_dict=state_dict, model=model)


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
