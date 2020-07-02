# Type hinting
from typing import Optional, Union
from pathlib import Path


class DummyRunLogger:
    """Utility class to placehold the logger."""
    def __init__(self):
        """Initialize the run-logger."""
        pass

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
