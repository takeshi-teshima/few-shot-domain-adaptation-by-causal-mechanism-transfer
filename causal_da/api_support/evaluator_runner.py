# Type hinting
from typing import Optional, Iterable
from causal_da.api_support.evaluator import EvaluatorBase


class EvaluatorRunner:
    """A utility to run a set of evaluators in an experiment run."""
    def __init__(self,
                 evaluators: Iterable[EvaluatorBase],                                          
                 run_interval: Optional[int] = None):
        """
        Parameters:
            evaluators: the list of evaluators.
            run_interval: the interval at which the evaluations should run in a training loop (unit: epoch).
        """
        self.evaluators = evaluators
        self.run_interval = run_interval

    def extend(self, evaluators: Iterable[EvaluatorBase]):
        """Add some evaluators.
        Parameters:
            evaluators: the evaluators to be added.
        """
        if isinstance(self.evaluators, list):
            self.evaluators + evaluators

            _res = self.evaluators.extend(evaluators)
        if _res is not None:
            self.evaluators = _res

    def __call__(self, epoch: Optional[int] = None):
        """Run the evaluations.

        Parameters:
            epoch: the epoch when this evaluation is run (``None`` if outside a training loop).
        """
        if (epoch is None) or self.is_epoch_to_run(epoch):
            for evaluator in self.evaluators:
                evaluator(epoch)

    def is_epoch_to_run(self, epoch: int) -> bool:
        """Decide whether the evaluations should run at this epoch.

        Parameters:
            epoch: the epoch when this evaluation is run (``None`` if outside a training loop).
        """
        if self.run_interval is None:
            return True
        elif (epoch % self.run_interval == 0):
            return True
        else:
            return False
