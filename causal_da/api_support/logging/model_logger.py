from pathlib import Path
import mlflow
import mlflow.pytorch
import shutil
import zipfile
from tempfile import TemporaryDirectory

# Type hinting
from typing import Optional
from causal_da.api_support.logging.base import RunLogger


class MLFlowModelSaver:
    """The base class of the model saver classes using ``mlflow.pytorch``."""
    def __init__(self, path: str):
        """
        Parameters:
            path: the path at which the model should be stored.
        """
        self.path = Path(path)

    def save(self, model):
        """Save the model using ``mlflow.pytorch.save_model``.

        Parameters:
            model: the ``pytorch`` model to be saved.
        """
        if self.path.exists():
            shutil.rmtree(self.path)
        mlflow.pytorch.save_model(model, str(self.path))

    def load(self):
        """Load the model using ``mlflow.pytorch.load_model``."""
        return mlflow.pytorch.load_model(str(self.path))

    @classmethod
    def load_zip(cls, zip_path: str):
        """Utility class method to load a zipped model using ``mlflow.pytorch.load_model``.

        Parameters:
            zip_path: path to the zipped model.
        """
        with TemporaryDirectory() as temp_dirname:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dirname)
                model = mlflow.pytorch.load_model(temp_dirname)
        return model


class MLFlowModelLogger(MLFlowModelSaver):
    """The model saver class using ``mlflow.pytorch``.
    This is more handy than ``MLFlowMultiModelLogger`` when we only save the model once.
    """
    def __init__(self,
                 path: str,
                 db_key: str,
                 run_logger: RunLogger,
                 compress_format: Optional[str] = None):
        """
        Parameters:
            path: the path where the model should be saved.
            db_key: the database key to save the model path.
            run_logger: the logger to save the experiment information.
            compress_format: the format of compression (``None``: default ``'zip'`` is used. Other options are ``('zip', 'tar', 'gztar', 'bztar', 'xztar')``).
        """
        super().__init__(path)
        self.run_logger = run_logger
        self.db_key = db_key
        if compress_format is None:
            compress_format, suffix = [('zip', '.zip'), ('tar', '.tar'),
                                       ('gztar', '.tar.gz'),
                                       ('bztar', '.tar.bz'),
                                       ('xztar', '.tar.xz')][0]
        self._format = compress_format
        self._suffix = suffix

    def save(self, model):
        """Save the model using ``mlflow.pytorch.save_model``.

        Parameters:
            model: the ``pytorch`` model to be saved.
        """
        import shutil
        super().save(model)
        zip_path = self.run_logger.artifact_temp_dir_path / self.path.stem
        shutil.make_archive(zip_path, self._format, root_dir=self.path)
        # self.run_logger.log_artifact(str(zip_path.with_suffix(self._suffix)), '')
        self.run_logger.set_tags(
            {self.db_key: str(zip_path.with_suffix(self._suffix).resolve())})

    def load(self):
        """Load the model using ``mlflow.pytorch.load_model``."""
        raise NotImplementedError()
        return mlflow.pytorch.load_model(str(self.path))


class MLFlowMultiModelLogger(MLFlowModelSaver):
    """The model saver class using ``mlflow.pytorch``.
    This is more handy than ``MLFlowModelLogger`` when we save the intermediate models during a training loop.
    """
    def __init__(self,
                 path: str,
                 db_key: str,
                 run_logger: RunLogger,
                 compress_format=None):
        """
        Parameters:
            path: the path where the model should be saved.
            db_key: the database key to save the model path.
            run_logger: the logger to save the experiment information.
            compress_format: the format of compression (``None``: default ``'zip'`` is used. Other options are ``('zip', 'tar', 'gztar', 'bztar', 'xztar')``).
        """
        super().__init__(path)
        self.run_logger = run_logger
        self.db_key = db_key
        if compress_format is None:
            compress_format, suffix = [('zip', '.zip'), ('tar', '.tar'),
                                       ('gztar', '.tar.gz'),
                                       ('bztar', '.tar.bz'),
                                       ('xztar', '.tar.xz')][0]
        self._format = compress_format
        self._suffix = suffix

    def _build_zip_path(self, root_path, path, _id, epoch):
        """Build the zip path using the epoch and experiment run id.

        Parameters:
            root_path: the base path where the model should be saved.
            path: the path describing the identifying name of the model (can be the same for different ``_id`` and ``epoch``).
            _id: the experiment run id.
            epoch: the epoch of the training.
        """
        return root_path / path.stem / str(_id) / str(epoch)

    def save(self, model, epoch):
        """Save the intermediate model of a training epoch.

        Parameters:
            model: the model to be saved.
            epoch: the epoch at which this method is called.
        """

        with TemporaryDirectory() as temp_dirname:
            path = Path(temp_dirname)
            if path.exists():
                shutil.rmtree(path)
            mlflow.pytorch.save_model(model, str(temp_dirname))
            zip_path = self._build_zip_path(
                self.run_logger.artifact_temp_dir_path, self.path,
                self.run_logger.get_current_run_id(), epoch)
            shutil.make_archive(zip_path, self._format, root_dir=temp_dirname)

        current_tags = self.run_logger.get_tags(self.db_key)
        if current_tags is None:
            current_tags = {}
        current_tags.update(
            {str(epoch): str(zip_path.with_suffix(self._suffix).resolve())})
        self.run_logger.set_tags({self.db_key: current_tags})
