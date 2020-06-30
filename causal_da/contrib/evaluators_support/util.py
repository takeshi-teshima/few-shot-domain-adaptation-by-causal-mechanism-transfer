from tempfile import TemporaryDirectory
from pathlib import Path


class ArtifactLoggable:
    def save_artifact(self, artifact_subdir, filename, save_fn):
        """
        save_fn (Callable) : A function that takes a path string (pointing to the filename to save) as input and saves a file.
        """
        with TemporaryDirectory() as temp_dirname:
            _path_base = Path(temp_dirname)
            _path_base.mkdir(parents=True, exist_ok=True)
            _path = str(_path_base / filename)
            save_fn(_path)
            self.run_logger.log_artifact(_path, artifact_subdir)

    def batch_save_artifact(self):
        raise NotImplementedError()
