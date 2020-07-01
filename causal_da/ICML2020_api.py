from pathlib import Path
from typing import Iterable, Dict, Optional, Union
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from .components.ica_torch.api import GCLTrainableInvertibleICAModel
from .components.inn_torch import GlowNonExponential
from .algorithm.api import CausalMechanismTransfer
from .api_support.index_class_label import index_class_label
from .api_support.logging import DummyRunLogger, MongoAndSacredRunLogger, ParamHistoryManagerBase
from .api_support.logging.model_logger import MLFlowModelLogger
from .components.aug_predictor import AugKRR


class _ICML2020API_Single_Run_Wrapper:
    """Perform a single run and record the results.
    This is essentially just its ``train_eval_single()`` method but with the ability to hold the unchanged arguments throughout all runs.
    """
    def __init__(self, static_args, static_kwargs):
        self.static_args = static_args
        self.static_kwargs = static_kwargs

    def __call__(self, idx, params_injector, run_logger):
        return self.train_eval_single(idx, params_injector, run_logger,
                                      *self.static_args, **self.static_kwargs)

    def train_eval_single(self,
                          idx,
                          params_injector,
                          run_logger,
                          X_src,
                          Y_src,
                          c_src,
                          save_model_path: Path,
                          ica_intermediate_evaluators,
                          augmenter_evaluators,
                          train_params=dict(epochs=500,
                                            batch_size=32,
                                            device='cpu',
                                            val_freq=100)):
        """Run Causal DA and record the performance for a single hyper-parameter setting.

        Parameters
        ----------
        idx : ``int``
            A unique ID of the experiment run. Used to identify the saved model file.
            Supposed to be the value generated by ``perform_run()`` of a ``MongoAndSacredRunLogger`` instance.

        params_injector

        run_logger

        X_src

        Y_src

        c_src

        save_model_path

        ica_intermediate_evaluators

        train_params : ``dict``
            Contains ``(epochs, batch_size, device, val_freq)`` as the keys.
        """
        data_src = np.hstack((X_src, Y_src))
        c_src = index_class_label(c_src)
        dim = data_src.shape[1]
        n_label = len(np.unique(c_src, return_index=True)[1])
        save_model_path = str(save_model_path / str(idx))

        # 0. Prepare recording.
        run_logger.start_run()
        best_score_model_logger = MLFlowModelLogger(save_model_path,
                                                    'best_score_model',
                                                    run_logger)

        # 1. Build the INN model to be used inside the ICA model.
        inn = self._get_inn(params_injector, dim, run_logger)

        # 2. Build the out-of-box ICA model to be passed to our method.
        invertible_ica_model = self._prepare_trainable_ica_model(
            inn, params_injector, dim, n_label, run_logger, train_params,
            save_model_path)

        # 3. Run the method and record the results.
        predictor_model = AugKRR()
        method = CausalMechanismTransfer(invertible_ica_model, predictor_model)
        ica_data = (data_src, c_src)
        ica_loggers = (run_logger, best_score_model_logger)

        method.train_and_record(data_src, ica_data, ica_loggers,
                                ica_intermediate_evaluators,
                                augmenter_evaluators)

    def _get_inn(self, params_injector, dim, run_logger):
        depth = params_injector.get('depth')
        n_hidden = params_injector.get('n_hidden')
        inn = GlowNonExponential(depth=depth, dim=dim, n_hidden=n_hidden)
        run_logger.log_params({
            'depth': depth,
            'n_hidden': n_hidden,
        })
        return inn

    def _prepare_trainable_ica_model(self, inn, params_injector, dim, n_label,
                                     run_logger, train_params,
                                     save_model_path):
        classifier_hidden_dim = params_injector.get('classifier_hidden_dim')
        classifier_n_layer = params_injector.get('classifier_n_layer')
        run_logger.log_params({
            'classifier_n_layer': classifier_n_layer,
            'classifier_hidden_dim': classifier_hidden_dim,
        })
        ica_model = GCLTrainableInvertibleICAModel(inn, dim,
                                                   classifier_hidden_dim,
                                                   n_label, classifier_n_layer)
        lr = params_injector.get('lr')
        weight_decay = params_injector.get('weight_decay')
        device = train_params['device']
        batch_size = train_params.get('batch_size')
        max_epochs = train_params.get('epochs')
        ica_model.set_train_params(lr, weight_decay, device, batch_size,
                                   max_epochs)
        return ica_model


class CausalMechanismTransferICML2020API:
    """The main interface class for the experiments. Based on the following combination of components:

    * Hyper-paramter search: grid-search.

      * HP selection is assumed to be performed by recording the validation scores for all runs in MongoDB and comparing the scores later.

    * Experiment recording: Sacred + MongoDB.

    * Model recording: MLFlow.

    * ICA

      * training: GCL (out-of-the-box).

      * Invertible neural network: Glow-based (without exponential activation).

    * Predictor

      * Kernel ridge regression (RBF kernel)
    """
    def __init__(
            self,
            parallel_split_index: int,
            total_parallel_split: int,
            run_logger: Optional[MongoAndSacredRunLogger] = None,
            param_history_manager: Optional[ParamHistoryManagerBase] = None):
        """Construct the method object.

        Parameters
        ----------
        parallel_split_index : ``int``
            We use brute-force parallelization such that we split the set of
            all hyperparameter configurations into ``total_parallel_split`` chunks.

        total_parallel_split : ``int``
            The total number of hyper-parameter splits for parallelization. See also ``parallel_split_index``.
        """
        self.parallel_split_index = parallel_split_index
        self.total_parallel_split = total_parallel_split
        if run_logger is None:
            self.run_logger = DummyRunLogger()
        else:
            self.run_logger = run_logger
        self.param_history_manager = param_history_manager

    def _prepare_param_grid(self, cfg_method):
        space = cfg_method['base_param']
        space.update(cfg_method['model_param'])
        param_grid = list(ParameterGrid(space))
        param_grid = self._split_and_filter_hyperparam_candidates(param_grid)
        return param_grid

    def _split_and_filter_hyperparam_candidates(self, param_grid):
        param_grid = np.array_split(
            param_grid,
            self.total_parallel_split)[self.parallel_split_index - 1]
        if self.param_history_manager is not None:
            param_grid = self.param_history_manager.filter(param_grid)
        return param_grid

    def run_method_and_eval(self,
                            X_src_train,
                            Y_src_train,
                            c_src_train,
                            cfg_method,
                            save_model_path: Path,
                            ica_intermediate_evaluators,
                            augmenter_evaluators,
                            run_kwargs,
                            debug: bool = False):
        """Main interface.
        Runs through all the hyper-paramter candidates except those for which we have a previous run record.
        Designed for training, probing the performance, and saving the results in a database that is connected to Sacred run database (via sacred's ``run_id``), for all hyper-parameter combinations.

        Parameters
        ----------
        X_src_train

        Y_src_train

        c_src_train

        cfg_method

        save_model_path : ``pathlib.Path``

        ica_intermediate_evaluators

        augmenter_evaluators

        run_kwargs

        debug : ``bool``
            Debug mode (default ``False``). If false, exceptions are not caught during the loop.
        """
        args = X_src_train, Y_src_train, c_src_train, save_model_path, ica_intermediate_evaluators, augmenter_evaluators
        self.single_run_wrapper = _ICML2020API_Single_Run_Wrapper(
            args, run_kwargs)
        param_grid = self._prepare_param_grid(cfg_method)
        for params in param_grid:
            if debug:
                self.run_logger.perform_run(
                    lambda idx, _params: self.single_run_wrapper(
                        idx, _params, run_logger=self.run_logger), params)
            else:
                try:
                    self.run_logger.perform_run(
                        lambda idx, _params: self.single_run_wrapper(
                            idx, _params, run_logger=self.run_logger), params)
                except Exception as err:
                    print(err)