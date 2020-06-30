class AugmenterEvaluator:
    def set_augmenter(self, augmenter):
        self.augmenter = augmenter

    def __call__(self):
        pass


class AugmenterEvaluators:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def set_augmenter(self, augmenter):
        for evaluator in self.evaluators:
            evaluator.set_augmenter(augmenter)
