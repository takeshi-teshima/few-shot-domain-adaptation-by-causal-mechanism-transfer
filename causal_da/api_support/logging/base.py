from abc import abstractmethod

# Type hinting
from typing import Iterable, Dict, Any, Union, Optional
from pathlib import Path


class RunLogger:
    """The base class of the run_loggers used to record the experiment setup/records."""
    @abstractmethod
    def start_run(self):
        """A callback called at the beginning of the experiment run."""
        raise NotImplementedError()

    @abstractmethod
    def end_run(self):
        """A callback called at the end of the experiment run."""
        raise NotImplementedError()

    @abstractmethod
    def log_params(self, params_dict: Dict[str, Any]):
        """Log parameters."""
        raise NotImplementedError()

    @abstractmethod
    def set_tags(self, tags_dict):
        """Log the tags."""
        raise NotImplementedError()

    @abstractmethod
    def get_tags(self, key: str) -> Any:
        """Get the tag value.

        Parameters:
            key: the key to access the tag value.
        """
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(self, dic: Dict[str, Any], step=None):
        """Log the metrics.

        Parameters:
            dic: the dictionary containing the values to be recorded.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_tags_exp_wide(self, tags_dict):
        """Log the experiment-wide tags."""
        raise NotImplementedError()

    @abstractmethod
    def log_params_exp_wide(self, params_dict):
        """Log the experiment-wide parameters."""
        raise NotImplementedError()

    @abstractmethod
    def log_artifact(self, _path: Union[str, Path], artifact_subdir: str):
        """Record an artifact (e.g., a model) that is already saved at the specified path.
        The behavior may differ among different run loggers.
        For example, this method may retrieve the data in ``_path`` and save it to another location under the sub-directory ``artifact_dir``.

        Parameters:
            _path: the path where the artifact has been saved.
            artifact_subdir: a string to specify under what sub-directory names the artifact should be stored in the database, remote storage, etc.
        """
        raise NotImplementedError()


class DummyRunLogger(RunLogger):
    """Utility class to placehold the logger."""
    def __init__(self):
        """Initialize the run-logger."""
        raise NotImplementedError()

    def start_run(self):
        """A callback called at the beginning of the experiment run."""
        pass

    def end_run(self):
        """A callback called at the end of the experiment run."""
        pass

    def log_params(self, params_dict: dict):
        """Log parameters."""
        pass

    def set_tags(self, tags_dict: dict):
        """Log the tags."""
        pass

    def log_metrics(self, dic: dict, step: Optional[int] = None):
        """Log the metrics."""
        pass

    def set_tags_exp_wide(self, tags_dict: dict):
        """Log the experiment-wide tags."""
        pass

    def log_params_exp_wide(self, params_dict: dict):
        """Log the experiment-wide parameters."""
        pass

    def log_artifact(self, _path: Union[str, Path], folder_name: str):
        """Log an artifact (e.g., a model)."""
        pass
