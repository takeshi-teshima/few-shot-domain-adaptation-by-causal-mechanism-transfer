from pathlib import Path
from sacred import Experiment
from typing import Union
import shutil
from .base import RunLogger
from mexlet.records.mongo import sanitize_data


def sanitize_data(r):
    n = {}
    for k, v in r.items():
        if isinstance(v, np.int64):
            v = int(v)
        if isinstance(v, np.float64):
            v = float(v)
        if isinstance(v, np.float32):
            v = float(v)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        n[k] = v
    return n


class MongoAndSacredRunLogger(RunLogger):
    def __init__(self,
                 experiment_name,
                 observer,
                 mongo_table,
                 artifact_temp_dir='sacred_artifact/1'):
        self.ex = Experiment(experiment_name)
        self.ex.observers.append(observer)
        self.artifact_temp_dir_path = Path(artifact_temp_dir)
        self.artifact_temp_dir_path.mkdir(parents=True, exist_ok=True)
        self.mongo_table = mongo_table

        # Placeholders
        self.exp_wide_info = {}
        self.exp_wide_artifacts = []
        self._run = None

    def start_run(self):
        pass

    def end_run(self):
        pass

    def perform_run(self, func, params):
        @self.ex.main
        def _main(_run):
            self._run = _run
            self.mongo_record_id = self.mongo_table.insert_one(
                sanitize_data({
                    **{
                        'sacred_run_id': self._run._id
                    },
                    **self.exp_wide_info,
                    **{
                        'exp_wide_artifacts': self.exp_wide_artifacts
                    },
                    **params
                })).inserted_id
            self._run.info.update(self.exp_wide_info)
            self._run.info.update(
                {'exp_wide_artifacts': self.exp_wide_artifacts})
            val = func(f"{self._run.experiment_info['name']}_{self._run._id}",
                       params)
            self.update_mongo({'finished': True})
            self._run = None
            self.mongo_record_id = None
            return val

        run = self.ex.run()
        return run.result

    def get_current_run_id(self):
        if self._run is not None:
            return self._run._id
        else:
            return None

    def update_mongo(self, dic):
        self.mongo_table.find_one_and_update({'_id': self.mongo_record_id},
                                             {'$set': sanitize_data(dic)})

    def log_params(self, params_dict):
        # self._run.config.update(params_dict)
        self._run.info.update(params_dict)
        self.update_mongo(params_dict)

    def set_tags(self, tags_dict):
        # self._run.config.update(tags_dict)
        self._run.info.update(tags_dict)
        self.update_mongo(tags_dict)

    def get_tags(self, key):
        return self._run.info.get(key)

    def set_tags_exp_wide(self, tags_dict):
        self.exp_wide_info.update(tags_dict)

    def log_params_exp_wide(self, params_dict):
        self.exp_wide_info.update(params_dict)

    def log_metrics(self, dic, step=None):
        for key, val in dic.items():
            if step is not None:
                self._run.log_scalar(key, val, step)
                # TODO: Implement logging to MongoDB.
            else:
                self._run.log_scalar(key, val)
                self._run.info[key] = val
                self.update_mongo({key: val})

    def log_artifact(self, _path: Union[str, Path], artifact_subdir: str):
        _path = Path(_path)
        if self._run is not None:
            self._run.add_artifact(_path,
                                   name='___'.join(
                                       [artifact_subdir, _path.name]))
            # for x in (_path / artifact_subdir).glob('**/*'):
            #     if x.is_dir():
            #         continue
            #     self._run.add_artifact(x, name=artifact_subdir + x)
            # self._run.add_resource(_path, artifact_subdir)
        else:
            (self.artifact_temp_dir_path / artifact_subdir).mkdir(
                parents=True, exist_ok=True)
            shutil.copy(str(_path),
                        str(self.artifact_temp_dir_path / artifact_subdir))
            self.exp_wide_artifacts.append(
                str(self.artifact_temp_dir_path / artifact_subdir /
                    _path.name))
