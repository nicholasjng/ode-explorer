import json
import os
from typing import Dict, Text, Any

from ode_explorer.constants import ResultKeys, ConfigKeys
# from ode_explorer.utils.data_utils import write_result_to_csv, convert_to_dict, initialize_dim_names


def get_result_metadata(config: Dict[Text, Any]):
    """
    Get metadata from a result.

    Args:
        config: Result config object saved in an Integrator instance.

    Returns:
        A dict with run metadata information.
    """

    metadata_keys = [ConfigKeys.TIMESTAMP, ConfigKeys.ID]
    metadata = {k: v for k, v in config.items() if k in metadata_keys}

    return metadata


def write_result_to_disk(result: Dict, out_dir: Text, **kwargs):
    """
    Save a run to disk, including result data, metrics and additional info.

    Args:
        result: Result object saved in an Integrator instance.
        out_dir: Designated output directory.
        **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_csv.
    """
    result_data = result.pop(ResultKeys.RESULT_DATA)
    result_config = result.pop(ResultKeys.CONFIG)

    result_filename = "result_info.json"

    # for i, res in enumerate(result_data):
    #     result_data[i] = convert_to_dict(res,
    #                                      model_metadata=model_metadata,
    #                                      dim_names=dim_names)

    # write result vectors to csv file
    # write_result_to_csv(result=result_data,
    #                     out_dir=out_dir,
    #                     outfile_name=ResultKeys.RESULT_DATA,
    #                     **kwargs)

    # write metrics to csv file
    # write_result_to_csv(result=metric_data,
    #                     out_dir=out_dir,
    #                     outfile_name=ResultKeys.METRICS,
    #                     **kwargs)

    outfile = os.path.join(out_dir, result_filename)
    with open(outfile, "w") as f:
        json.dump(result_config, f)
