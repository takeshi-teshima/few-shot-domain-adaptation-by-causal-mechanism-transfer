class EvaluatorRunner:
    def __init__(self, evaluators, run_interval=None):
        self.evaluators = evaluators
        self.run_interval = run_interval

    def extend(self, evaluators):
        if isinstance(self.evaluators, list):
            self.evaluators + evaluators

            _res = self.evaluators.extend(evaluators)
        if _res is not None:
            self.evaluators = _res

    def __call__(self, epoch=None):
        if (epoch is None) or self.is_epoch_to_run(epoch):
            for evaluator in self.evaluators:
                evaluator(epoch)

    def is_epoch_to_run(self, epoch) -> bool:
        if self.run_interval is None:
            return True
        elif (epoch % self.run_interval == 0):
            return True
        else:
            return False


class DummyEvaluatorRunner(EvaluatorRunner):
    def __init__(self):
        pass

    def __call__(self, augmenter, epoch=None):
        pass

    def is_epoch_to_run(self, epoch) -> bool:
        return False
