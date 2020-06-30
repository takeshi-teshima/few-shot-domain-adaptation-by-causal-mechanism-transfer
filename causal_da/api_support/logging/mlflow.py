from .base import RunLogger
import mlflow


class MLFlowRunLogger(RunLogger):
    def __init__(self, mlflow_tracking_uri, experiment_name, data_run_id):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=data_run_id)

    def start_run(self):
        mlflow.start_run(nested=True)

    def end_run(self):
        mlflow.end_run()

    def log_params(self, params_dict):
        mlflow.log_params(params_dict)

    def set_tags(self, tags_dict):
        mlflow.set_tags(tags_dict)

    def set_tags_exp_wide(self, tags_dict):
        """TODO: This runs even if the task is not nested. Make sure the global tag setter does not run inside of a run."""
        mlflow.set_tags(tags_dict)

    def log_params_exp_wide(self, params_dict):
        """TODO: This runs even if the task is not nested. Make sure the global tag setter does not run inside of a run."""
        mlflow.log_params(params_dict)

    def log_metrics(self, dic, step=None):
        mlflow.log_metrics(dic, step)

    def log_artifact(self, _path, folder_name):
        mlflow.log_artifact(_path, folder_name)
