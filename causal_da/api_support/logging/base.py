from abc import abstractmethod
from typing import Iterable


class RunLogger:
    @abstractmethod
    def start_run(self):
        raise NotImplementedError()

    @abstractmethod
    def end_run(self):
        raise NotImplementedError()

    @abstractmethod
    def log_params(self, params_dict):
        raise NotImplementedError()

    @abstractmethod
    def set_tags(self, tags_dict):
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(self, dic, step=None):
        raise NotImplementedError()

    @abstractmethod
    def set_tags_exp_wide(self, tags_dict):
        raise NotImplementedError()

    @abstractmethod
    def log_params_exp_wide(self, params_dict):
        raise NotImplementedError()

    @abstractmethod
    def log_artifact(self, _path, folder_name):
        raise NotImplementedError()


class JointRunLogger(RunLogger):
    def __init__(self, *loggers: Iterable[RunLogger]):
        self.loggers = loggers

    def start_run(self):
        for logger in self.loggers:
            logger.start_run()

    def end_run(self):
        for logger in self.loggers:
            logger.end_run()

    def log_params(self, params_dict):
        for logger in self.loggers:
            logger.log_params(params_dict)

    def set_tags(self, tags_dict):
        for logger in self.loggers:
            logger.set_tags(tags_dict)

    def set_tags_exp_wide(self, tags_dict):
        for logger in self.loggers:
            logger.set_tags_exp_wide(tags_dict)

    def log_params_exp_wide(self, params_dict):
        for logger in self.loggers:
            logger.log_params_exp_wide(params_dict)

    def log_metrics(self, dic, step=None):
        for logger in self.loggers:
            logger.log_metrics(dic, step)

    def log_artifact(self, _path, folder_name):
        for logger in self.loggers:
            logger.log_artifact(_path, folder_name)


class DummyRunLogger(RunLogger):
    def __init__(self):
        raise NotImplementedError()

    def start_run(self):
        pass

    def end_run(self):
        pass

    def log_params(self, params_dict):
        pass

    def set_tags(self, tags_dict):
        pass

    def log_metrics(self, dic, step=None):
        pass

    def set_tags_exp_wide(self, tags_dict):
        pass

    def log_params_exp_wide(self, params_dict):
        pass

    def log_artifact(self, _path, folder_name):
        pass
