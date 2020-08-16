import pandas as pd
import os
import datetime
from typing import List, Dict, Text


def write_results_to_file(result_data: List[Dict[Text, float]],
                          out_dir: Text,
                          out_name: Text = None,
                          **kwargs) -> None:
    """
    Small wrapper to write ODE simulation results to a .csv file.
    :param result_data: Result data to be written to file.
    :param out_dir: Log output directory
    :param out_name: Log output file name.
    :param kwargs: Additional keyword arguments, passed to pandas.to_csv.
    :return: None.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    file_ext = ".csv"

    # convert result_list to data frame
    # fast, construction from list with schema dict
    result_df = pd.DataFrame(data=result_data)

    out_name = out_name or datetime.datetime.now().strftime('%Y-%m-%d')

    out_file = os.path.join(out_dir, out_name)

    result_df.to_csv(out_file + file_ext, **kwargs)
