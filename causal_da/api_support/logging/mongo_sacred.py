from pathlib import Path
from sacred import Experiment
import shutil
from .base import RunLogger
import numpy as np

# Type hinting
from typing import Union, Callable, Any, Dict, Optional
from pymongo.collection import Collection
from sacred.observers.mongo import MongoObserver


def sanitize_data(r):
    """Sanitize the data values to cast to the types that can be handled by MongoDB (via PyMongo)."""
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
    """A run logger based on MongoDB and Sacred."""
    def __init__(self,
                 experiment_name: str,
                 observer: MongoObserver,
                 mongo_table: Collection,
                 artifact_temp_dir: str = 'sacred_artifact/1'):
        """
        Parameters:
            experiment_name: the name of the experiment.
            observer: the observer object.
            mongo_table: the table object of PyMongo.
            artifact_temp_dir: the path to the directory that can be used to store the artifacts temporarily.
        """
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
        """A callback called at the beginning of the experiment run."""
        pass

    def end_run(self):
        """A callback called at the end of the experiment run."""
        pass

    def perform_run(self, func: Callable[[str, Any], Any], params: Any):
        """The method to perform an experiment.

        Parameters:
            func: a function of two variables: ``(idx, params)``.
                  ``idx`` is a string indicating the unique identifier of the experiment run
                  that can be used in the function, e.g., to save artifacts with names that are distinct among different runs.
            params: the parameters to be passed to ``func``.

        Note:
            Here, we capsule the experiment starting procedure of Sacred for the convenience of the user.
        """
        @self.ex.main
        def _main(_run):
            """The local method defined to wrap the run for interfacing with Sacred."""
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
        """Obtain the current Sacred run_id."""
        if self._run is not None:
            return self._run._id
        else:
            return None

    def update_mongo(self, dic: Dict[str, Any]) -> None:
        """Update a dictionary to the MongoDB.

        Parameters:
            dic: dictionary containing the key-value pairs to be updated.
        """
        self.mongo_table.find_one_and_update({'_id': self.mongo_record_id},
                                             {'$set': sanitize_data(dic)})

    def log_params(self, params_dict: Dict[str, Any]) -> None:
        """Log parameters."""
        # self._run.config.update(params_dict)
        self._run.info.update(params_dict)
        self.update_mongo(params_dict)

    def set_tags(self, tags_dict: Dict[str, Any]) -> None:
        """Log the tags."""
        # self._run.config.update(tags_dict)
        self._run.info.update(tags_dict)
        self.update_mongo(tags_dict)

    def get_tags(self, key: str) -> Any:
        """Get the tag value.

        Parameters:
            key: the key to access the tag value.
        """
        return self._run.info.get(key)

    def set_tags_exp_wide(self, tags_dict: Dict[str, Any]):
        """Log the experiment-wide tags."""
        self.exp_wide_info.update(tags_dict)

    def log_params_exp_wide(self, params_dict: Dict[str, Any]):
        """Log the experiment-wide parameters."""
        self.exp_wide_info.update(params_dict)

    def log_metrics(self,
                    dic: Dict[str, Union[float, int]],
                    step: Optional[int] = None):
        """Log the metrics.

        Parameters:
            dic: the dictionary containing the values to be recorded.
        """
        for key, val in dic.items():
            if step is not None:
                self._run.log_scalar(key, val, step)
                # TODO: Implement logging to MongoDB.
            else:
                self._run.log_scalar(key, val)
                self._run.info[key] = val
                self.update_mongo({key: val})

    def log_artifact(self, _path: Union[str, Path], artifact_subdir: str):
        """Record an artifact (e.g., a model) that is already saved at the specified path.

        Parameters:
            _path: the path where the artifact has been saved.
            artifact_subdir: a string to specify under what sub-directory names the artifact should be stored in the database, remote storage, etc.
        """
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
