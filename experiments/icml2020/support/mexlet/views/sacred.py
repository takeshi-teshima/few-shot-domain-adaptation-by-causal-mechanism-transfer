from typing import Callable, Optional
import numpy as np
import pandas as pd
from pandas_ply import install_ply, X
from .pandas import pd_add_column
install_ply(pd)
import copy


def _select_first(a, b, b_val):
    _v = np.nonzero(b == b_val)[0]
    if len(_v) == 0:
        return np.nan
    return a[_v[0]]


def select_by_argmin(metrics,
                     test_metric,
                     selector,
                     namer: Optional[Callable] = None):
    metrics = copy.copy(metrics)
    argmin = (metrics.ply_select(
        '*',
        argmin=lambda x: x['values'].map(lambda v: np.argmin(v)
                                         if len(v) != 0 else [])))
    argmin['argmin_step'] = argmin.apply(lambda x: x['steps'][x['argmin']],
                                         axis=1)

    # selector_df = argmin.ply_where(X.name == selector)
    combined = (pd_add_column(
        metrics.ply_where(X.name == test_metric),
        argmin.ply_where(X.name == selector).drop('name', axis=1).drop(
            ['values', 'steps'], axis=1), 'run_id',
        'run_id').ply_where(X.name == test_metric))
    if namer is None:
        namer = lambda name: 'min ' + name
    combined[namer(test_metric)] = combined.apply(
        lambda x: _select_first(x['values'], x['steps'], x['argmin_step']),
        axis=1)
    combined = combined.drop('values', axis=1).drop('argmin_step',
                                                    axis=1).drop('name',
                                                                 axis=1)
    return combined


def select_by_quantile(q,
                       metrics,
                       test_metric,
                       selector,
                       namer: Optional[Callable] = None,
                       interpolation='lower'):
    def _mapper(v):
        quantile = np.nanquantile(v, q, interpolation=interpolation)
        if np.isnan(quantile):
            return -1
        else:
            return int(v.index(quantile))

    argmin = (metrics.ply_select('*',
                                 argmin=lambda x: x['values'].map(_mapper)))

    # selector_df = argmin.ply_where(X.name == selector)
    combined = (pd_add_column(
        metrics.ply_where(X.name == test_metric),
        argmin.ply_where(X.name == selector).drop('name',
                                                  axis=1).drop('values',
                                                               axis=1),
        'run_id', 'run_id').ply_where(X.name == test_metric))
    if namer is None:
        namer = lambda name: 'min ' + name
    combined[namer(test_metric)] = combined.apply(
        lambda x: x['values'][x['argmin']] if x['argmin'] != -1 else np.nan,
        axis=1)
    combined = combined.drop('values', axis=1).drop('argmin',
                                                    axis=1).drop('name',
                                                                 axis=1)
    return combined


from mexlet.jupyter import print_df


def cleanse_nans_from_metrics(metrics, test_name, selector_name):
    """
    Removes NaNs from ``test_name`` metric as well as ``selector_name`` metric.
    The `'steps'` and the `'values'` are cleansed based on the values of ``test_name`` metric.
    """
    # Find removed_steps
    metrics['removed_steps'] = pd.DataFrame.from_dict(
        (metrics.ply_where(X.name == test_name).apply(
            lambda x: np.array(x['steps'])[np.isnan(x['values'])], axis=1)))
    removed_steps = metrics.ply_where(X.name == test_name)
    removed_steps = dict(
        zip(removed_steps['run_id'], removed_steps['removed_steps']))
    # Copy removed_steps to selector rows
    selector_steps = metrics.ply_where(X.name == selector_name).apply(
        lambda x: np.array(x['steps'])[~np.isin(np.array(x['steps']),
                                                removed_steps[x['run_id']])],
        axis=1)
    selector_values = metrics.ply_where(X.name == selector_name).apply(
        lambda x: np.array(x['values'])[~np.isin(np.array(x['steps']),
                                                 removed_steps[x['run_id']])],
        axis=1)
    test_steps = metrics.ply_where(X.name == test_name).apply(
        lambda x: np.array(x['steps'])[~np.isin(np.array(x['steps']),
                                                removed_steps[x['run_id']])],
        axis=1)
    test_values = metrics.ply_where(X.name == test_name).apply(
        lambda x: np.array(x['values'])[~np.isin(np.array(x['steps']),
                                                 removed_steps[x['run_id']])],
        axis=1)
    metrics.loc[metrics.name == selector_name, 'steps'] = selector_steps
    metrics.loc[metrics.name == selector_name, 'values'] = selector_values
    metrics.loc[metrics.name == test_name, 'steps'] = test_steps
    metrics.loc[metrics.name == test_name, 'values'] = test_values
    # Remove steps and values based on removed_steps
    metrics = metrics.drop('removed_steps', axis=1)
    return metrics
