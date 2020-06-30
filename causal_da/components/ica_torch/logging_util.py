class DummyRunLogger:
    def __init__(self):
        pass

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
