# dict formats, used for writing and displaying ODE integration data
class DataFormatKeys:
    ZIPPED = "zipped"
    VARIABLES = "variables"


# dynamic (step size integration) variables)
class DynamicVariables:
    MAX_STEPS = 10000
    INITIAL_H = 0.01


class RunKeys:
    RESULT_DATA = "result_data"
    METRICS = "metrics"
    RUN_METADATA = "run_metadata"
    RUN_CONFIG = "run_config"
    MODEL_METADATA = "model_metadata"


class RunConfigKeys:
    START = "start"
    END = "end"
    NUM_STEPS = "num_steps"
    STEP_SIZE = "h"
    METRIC_NAMES = "metric_names"
    CALLBACK_NAMES = "callback_names"


class RunMetadataKeys:
    TIMESTAMP = "timestamp"
    STEPFUNC_OUTPUT_FORMAT = "stepfunc_output_format"
    RUN_ID = "run_id"


class ModelMetadataKeys:
    DIM_NAMES = "dim_names"
    VARIABLE_NAMES = "variable_names"
    INDEP_NAME = "indep_name"
