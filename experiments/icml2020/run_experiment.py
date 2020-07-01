"""
Entry point for the experiment for ICML 2020.
Instantiate data, method, etc. and pass them to the method APIs which run the method and probe the performances.
"""
import os
from pathlib import Path
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
from causal_da.ICML2020_api import CausalMechanismTransferICML2020API
from causal_da.api_support.logging import MongoAndSacredRunLogger, PandasParamHistoryManager
from causal_da.api_support.validator.base import DummyValidationScorer
from causal_da.api_support.evaluator_runner import EvaluatorRunner, DummyEvaluatorRunner

from support.database.records_aggregator import MongoAggregator
from support.data import get_or_find_cached_data
from support.evaluate import get_epoch_evaluator_runner, get_augmenter_evaluators
from support.database.mongo import get_mongo_observer, get_table

# Importing this here will enable loading the dataset-wise module on the fly.
import data


def _evaluate_proposed_method(X_src, Y_src, c_src, tar_tr, tar_te, cfg,
                              run_logger):
    run_logger.set_tags_exp_wide(
        dict(
            gpu=cfg.misc.gpu,
            max_epochs=cfg.method.ica_train.max_epochs,
            batch_size=cfg.method.ica_train.batch_size,
            log_interval=cfg.recording.log_interval,
        ))

    run_logger.log_params_exp_wide({'src_train_size': len(X_src)})

    ######
    ## Prepare evaluators
    ##
    ica_intermediate_evaluators = get_epoch_evaluator_runner(
        cfg.recording.log_interval, run_logger)
    augmenter_evaluators = get_augmenter_evaluators(tar_tr, tar_te,
                                                    cfg.method.augment_size,
                                                    run_logger)

    ######
    ## Build and run methods
    ##
    save_model_path = Path(cfg.recording.save_model_path)
    run_kwargs = dict(
        train_params=dict(val_freq=cfg.recording.log_interval,
                          epochs=cfg.method.ica_train.max_epochs,
                          batch_size=cfg.method.ica_train.batch_size,
                          device="cuda" if cfg.misc.gpu else "cpu"))
    previous_results = MongoAggregator(
        get_table(cfg.recording.table, cfg.database.mongo_host,
                  cfg.database.mongo_port, cfg.database.mongo_user,
                  cfg.database.mongo_pass, cfg.database.mongo_dbname),
        query={
            'method': cfg.method.name,
            'data': cfg.data.name,
            'data_run_id': cfg.parallelization.data_run_id,
            'recording_set': cfg.recording.recording_set
        }).get_results_pd(index=None)

    method_experiment_api = CausalMechanismTransferICML2020API(
        cfg.method.run_split_number,
        cfg.method.run_split_total_number,
        run_logger=run_logger,
        param_history_manager=PandasParamHistoryManager(previous_results),
    )

    method_experiment_api.run_method_and_eval(
        X_src, Y_src, c_src, OmegaConf.to_container(cfg.method),
        save_model_path, ica_intermediate_evaluators, augmenter_evaluators,
        run_kwargs, cfg.debug)


@hydra.main(config_path='config/config.yaml')
def main(cfg: DictConfig):
    data = cfg.data.name
    data_path = hydra.utils.to_absolute_path(cfg.data.path)
    max_threads = cfg.misc.max_threads
    mlflow_tracking_uri = cfg.recording.mlflow_tracking_uri
    data_run_id = cfg.parallelization.data_run_id
    mlflow_tracking_uri = cfg.recording.mlflow_tracking_uri

    cfg.method.augment_size = eval(cfg.method.augment_size)

    mongo_params = cfg.database.mongo_host, cfg.database.mongo_port, cfg.database.mongo_user, cfg.database.mongo_pass, cfg.database.mongo_dbname
    run_logger = MongoAndSacredRunLogger(
        'icml2020', get_mongo_observer(*mongo_params),
        get_table(cfg.recording.table, *mongo_params),
        f'{cfg.recording.sacred_artifact_dir}/icml2020_{cfg.recording.recording_set}'
    )
    run_logger.set_tags_exp_wide(
        dict(data=data,
             data_path=data_path,
             data_run_id=data_run_id,
             max_threads=max_threads,
             mlflow_tracking_uri=mlflow_tracking_uri,
             recording_set=cfg.recording.recording_set,
             method=cfg.method.name))
    ######
    ## General config
    ##
    if max_threads != '-1':
        os.environ["OMP_NUM_THREADS"] = max_threads
    ######

    ######
    ## Get data
    ##
    if cfg.data.target_domain is not None:
        data_cache_name = f'{data}_{data_run_id}_{cfg.data.target_domain}'
    else:
        data_cache_name = f'{data}_{data_run_id}'

    data_module = import_module(f'data.{data}')
    dataset = data_module.get_data(cfg.data.path)

    (X_src, Y_src, c_src, X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, c_tar_tr,
     c_tar_te), target_c = get_or_find_cached_data(
         dataset,
         data_cache_name,
         target=cfg.data.target_domain,
         n_target_c=1,
         path=data_path,
         return_target_c=True)
    run_logger.set_tags_exp_wide({'Data type': data, 'target_c': target_c})

    tar_tr, tar_te = (X_tar_tr, Y_tar_tr, c_tar_tr), (X_tar_te, Y_tar_te,
                                                      c_tar_te)
    target_train_size = int(len(X_tar_tr) / len(target_c))
    run_logger.log_params_exp_wide({
        'target_train_size':
        target_train_size,
        'target_test_size':
        len(X_tar_te) / len(target_c)
    })

    run_logger.set_tags_exp_wide(
        {'Augmentation size': cfg.method.augment_size})
    # data,
    # data_run_id,
    # cfg.recording.recording_set, cfg.misc.n_trials,
    _evaluate_proposed_method(X_src, Y_src, c_src, tar_tr, tar_te, cfg,
                              run_logger)


if __name__ == '__main__':
    main()
