from pathlib import Path
import mlflow
import mlflow.pytorch
import shutil
from causal_da.api_support.logging.base import RunLogger
import zipfile
from tempfile import TemporaryDirectory


class MLFlowModelSaver:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, model):
        if self.path.exists():
            shutil.rmtree(self.path)
        mlflow.pytorch.save_model(model, str(self.path))

    def load(self):
        return mlflow.pytorch.load_model(str(self.path))

    @classmethod
    def load_zip(cls, zip_path):
        with TemporaryDirectory() as temp_dirname:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dirname)
                model = mlflow.pytorch.load_model(temp_dirname)
        return model


class MLFlowModelLogger(MLFlowModelSaver):
    def __init__(self,
                 path: str,
                 db_key: str,
                 run_logger: RunLogger,
                 compress_format=None):
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
        import shutil
        super().save(model)
        zip_path = self.run_logger.artifact_temp_dir_path / self.path.stem
        shutil.make_archive(zip_path, self._format, root_dir=self.path)
        # self.run_logger.log_artifact(str(zip_path.with_suffix(self._suffix)), '')
        self.run_logger.set_tags(
            {self.db_key: str(zip_path.with_suffix(self._suffix).resolve())})

    def load(self):
        raise NotImplementedError()
        return mlflow.pytorch.load_model(str(self.path))


class MLFlowMultiModelLogger(MLFlowModelSaver):
    def __init__(self,
                 path: str,
                 db_key: str,
                 run_logger: RunLogger,
                 compress_format=None):
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
        return root_path / path.stem / str(_id) / str(epoch)

    def save(self, model, epoch):

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
