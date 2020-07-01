import pandas as pd
from .aggregator import PdAggregator


def _select(r, cv_group, criterion, options):
    if options['larger_is_better']:
        idx = r.groupby(cv_group)[criterion].transform(max) == r[criterion]
    else:
        idx = r.groupby(cv_group)[criterion].transform(min) == r[criterion]
    return r.loc[idx, :]


def _select_one(r, cv_group, criterion, options):
    if options['larger_is_better']:
        idx = r.groupby(cv_group)[criterion].transform(max) == r[criterion]
    else:
        idx = r.groupby(cv_group)[criterion].transform(min) == r[criterion]
    return r.loc[idx, :].groupby(cv_group, as_index=False).nth(0)


class VirtualValidation:
    def __init__(self, aggregator, select_one=True):
        if isinstance(aggregator, pd.DataFrame):
            aggregator = PdAggregator(aggregator)
        self.aggregator = aggregator
        self.select_one = select_one

    def fit(self, cv_group, criteria):
        """
        CV by criteria
        If there are multiple rows, then second_criteria is used to search.
        """
        results_history = self.aggregator.get_results_pd()

        res = pd.DataFrame()
        for criterion_pair in criteria:
            if isinstance(criterion_pair[0], tuple):
                # If two criteria are given in the form of (('criterion_name', {'larger_is_better': True}), ('criterion2_name', {'larger_is_better': True}))
                criterion, options = criterion_pair[0]
                second_criterion, second_options = criterion_pair[1]
                if self.select_one:
                    rows = _select_one(results_history, cv_group, criterion, options)
                else:
                    rows = _select(results_history, cv_group, criterion, options)
                    rows = _select(rows, cv_group, second_criterion, second_options)
                rows.loc[:, 'HP_selected_by'] = criterion
                # rows_to_add = r[idx].groupby(cv_group).apply(
                #     lambda x: x.sample(n=1)).reset_index(drop=True)
                res = res.append(rows)
            else:
                # If only one criterion is given in the form of ('criterion_name', {'larger_is_better': True})
                criterion, options = criterion_pair
                if self.select_one:
                    rows = _select_one(results_history, cv_group, criterion, options)
                else:
                    rows = _select(results_history, cv_group, criterion, options)
                rows.loc[:, 'HP_selected_by'] = criterion
                res = res.append(rows)
        return res
