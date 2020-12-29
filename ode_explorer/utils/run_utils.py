import json
import os
import copy
from typing import Dict, Text

from ode_explorer import constants
from ode_explorer.constants import RunKeys, ModelMetadataKeys
from ode_explorer.utils.data_utils import write_result_to_csv, convert_to_dict, initialize_dim_names


def get_run_metadata(run):
    """
    Get metadata from a run.

    Args:
        run: Run object saved in an Integrator instance.

    Returns:
        A dict with run metadata information.
    """

    metadata_keys = [constants.TIMESTAMP, constants.RUN_ID]
    metadata = {k: v for k, v in run.items() if k in metadata_keys}

    return metadata


def write_run_to_disk(run: Dict, out_dir: Text, **kwargs):
    """
    Save a run to disk, including result data, metrics and additional info.

    Args:
        run: Run object saved in an Integrator instance.
        out_dir: Designated output directory.
        **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_csv.
    """

    run_copy = copy.deepcopy(run)

    result_data = run_copy.pop(RunKeys.RESULT_DATA)

    metric_data = run_copy.pop(RunKeys.METRICS)

    run_filename = "run_info.json"

    model_metadata = run_copy[RunKeys.MODEL_METADATA]

    variable_names = model_metadata[ModelMetadataKeys.VARIABLE_NAMES]

    dim_names = model_metadata[ModelMetadataKeys.DIM_NAMES]

    if not dim_names:
        dim_names = initialize_dim_names(variable_names, result_data[0])

    for i, res in enumerate(result_data):
        result_data[i] = convert_to_dict(res, model_metadata=model_metadata,
                                         dim_names=dim_names)

    # write result vectors to csv file
    write_result_to_csv(result=result_data,
                        out_dir=out_dir,
                        outfile_name=RunKeys.RESULT_DATA,
                        **kwargs)

    # write metrics to csv file
    write_result_to_csv(result=metric_data,
                        out_dir=out_dir,
                        outfile_name=RunKeys.METRICS,
                        **kwargs)

    outfile = os.path.join(out_dir, run_filename)
    with open(outfile, "w") as f:
        json.dump(run_copy, f)
