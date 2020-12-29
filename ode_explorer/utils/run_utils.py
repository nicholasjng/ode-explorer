import json
import os
from typing import Dict, Text

from ode_explorer import constants
from ode_explorer.constants import RunKeys
from ode_explorer.utils.data_utils import write_result_to_csv, convert_to_dict


def get_run_metadata(run):
    metadata_keys = [constants.TIMESTAMP, constants.RUN_ID]
    metadata = {k: v for k, v in run.items() if k in metadata_keys}

    return metadata


def write_run_to_disk(run: Dict, out_dir: Text):
    result_data = run[RunKeys.RESULT_DATA]

    metric_data = run[RunKeys.METRICS]

    run_filename = "run_info.json"

    model_metadata = run[RunKeys.MODEL_METADATA]

    for i, res in enumerate(result_data):
        result_data[i] = convert_to_dict(res, model_metadata=model_metadata)

    # write result vectors to csv file
    write_result_to_csv(result=result_data,
                        out_dir=out_dir,
                        outfile_name=RunKeys.RESULT_DATA)

    # write metrics to csv file
    write_result_to_csv(result=metric_data,
                        out_dir=out_dir,
                        outfile_name=RunKeys.METRICS)

    outfile = os.path.join(out_dir, run_filename)
    with open(outfile, "w") as f:
        json.dump(run, f)
