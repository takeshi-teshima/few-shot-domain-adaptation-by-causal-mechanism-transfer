import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '.'))
from pathlib import Path
import copy
from .util import Pickler


def get_or_find_cached_data(dataset, data_cache_name, **kwargs):
    if kwargs.get('target', None) is None:
        ## Target: randomly selected
        # res = SacredPickle(f'{data}_{data_run_id}').find_or_create(
        #     _run, lambda: dataset.load_stratified_targets(**kwargs))
        del kwargs['target']
        res = Pickler(data_cache_name, Path('pickle/data')).find_or_create(
            lambda: dataset.load_stratified_targets(**kwargs))
    else:
        ## Target: specified
        kwargs = copy.copy(kwargs)
        kwargs['target_c'] = [kwargs['target']]
        del kwargs['target']
        del kwargs['n_target_c']
        del kwargs['return_target_c']
        res = Pickler(data_cache_name, Path('pickle/data')).find_or_create(
            lambda: dataset.load_src_stratified_trg(**kwargs))
        res = res, kwargs['target_c']
    return res
