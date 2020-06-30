from abc import abstractmethod
from typing import Iterable, Dict, Any


class ParamHistoryManagerBase:
    @abstractmethod
    def filter(self, param_grid) -> Iterable[Dict]:
        """Given a set of candidate parameter sets, this function removes all existing sets
        and leaves the parameter set that is unexistent in the database of the previous runs.
        The set of keys appearing in its database needs to be a superset of the keys apperaing in the parameter candidates.
        """
        pass


class PandasParamHistoryManager(ParamHistoryManagerBase):
    def __init__(self, df):
        self.df = df

    def _df_has_value_set(self, df, values: Dict[str, Any]) -> bool:
        """Whether the df has a set of values.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict([{'a': 11, 'b': 12}, {'a': 21, 'b': 22}])
        >>> PandasParamHistoryManager(df)._df_has_value_set(df, {'a': 11, 'b': 12})
        True
        """
        if len(df) == 0:
            return False
        return len(
            df.query(' & '.join(
                [f'{key} == {val}' for key, val in values.items()]))) > 0

    def filter(self, param_grid) -> Iterable[Dict]:
        """
        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict([{'a': 11, 'b': 12, 'c': 10}])
        >>> param_grid = [{'a': 11, 'b': 12, 'c': 10}, {'a': 21, 'b': 22, 'c': 10}]
        >>> PandasParamHistoryManager(df).filter(param_grid)
        [{'a': 21, 'b': 22, 'c': 10}]

        # Considers the parameter to be existent even if the candidate contains only a subset of the values.
        # This assumes that no removal of the keys occurs along the development of the method.
        >>> PandasParamHistoryManager(df).filter([{'a': 11, 'b': 12}])
        []
        """
        res = []
        for param in param_grid:
            if not self._df_has_value_set(self.df, param):
                res.append(param)
        return res


if __name__ == '__main__':
    import doctest
    doctest.testmod()
