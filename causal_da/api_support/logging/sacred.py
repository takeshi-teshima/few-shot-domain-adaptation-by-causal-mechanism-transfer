from pathlib import Path
from sacred import Experiment
import shutil
from .base import RunLogger


class SacredRunLogger(RunLogger):
    def __init__(self, experiment_name, observer, artifact_temp_dir='sacred_artifact/1'):
        self.ex = Experiment(experiment_name)
        self.ex.observers.append(observer)
        self.artifact_temp_dir_path = Path(artifact_temp_dir)
        self.artifact_temp_dir_path.mkdir(parents=True, exist_ok=True)

        # Placeholders
        self.exp_wide_info = {}
        self.exp_wide_artifacts = []
        self._run = None

    def start_run(self):
        pass

    def end_run(self):
        pass

    def perform_run(self, func, params):
        # @self.ex.config_hook
        # def update_config(config, command_name, logger):
        #     config.update(params)
        #     return config
        @self.ex.main
        def _main(_run):
            self._run = _run
            self._run.info.update(self.exp_wide_info)
            self._run.info.update({'exp_wide_artifacts': self.exp_wide_artifacts})
            return_val = func(f"{self._run.experiment_info['name']}_{self._run._id}", params)
            self._run = None
            return return_val

        return self.ex.run()

    def log_params(self, params_dict):
        # self._run.config.update(params_dict)
        self._run.info.update(params_dict)

    def set_tags(self, tags_dict):
        # self._run.config.update(tags_dict)
        self._run.info.update(tags_dict)

    def set_tags_exp_wide(self, tags_dict):
        self.exp_wide_info.update(tags_dict)

    def log_params_exp_wide(self, params_dict):
        self.exp_wide_info.update(params_dict)

    def log_metrics(self, dic, step=None):
        for key, val in dic.items():
            if step is not None:
                self._run.log_scalar(key, val, step)
            else:
                self._run.log_scalar(key, val)

    def log_artifact(self, _path, artifact_subdir):
        _path = Path(_path)
        if self._run is not None:
            # self._run.add_artifact(_path, artifact_subdir)
            self._run.add_resource(_path, artifact_subdir)
        else:
            (self.artifact_temp_dir_path / artifact_subdir).mkdir(parents=True, exist_ok=True)
            shutil.copy(str(_path), str(self.artifact_temp_dir_path / artifact_subdir))
            self.exp_wide_artifacts.append(
                str(self.artifact_temp_dir_path / artifact_subdir / _path.name))
