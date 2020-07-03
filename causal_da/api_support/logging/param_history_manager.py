from abc import abstractmethod

# Type hinting
from typing import Iterable, Dict, Any, List
from pandas import DataFrame


class ParamHistoryManagerBase:
    """The base class of a parameter history manager."""
    @abstractmethod
    def filter(self, param_grid: List[Dict[str, Any]]) -> Iterable[Dict]:
        """Given a set of candidate parameter sets, this function removes all existing sets
        and leaves the parameter set that is unexistent in the database of the previous runs.
        The set of keys appearing in its database needs to be a superset of the keys apperaing in the parameter candidates.

        Parameters:
            param_grid: the list of records to be filtered in.
        """
        pass


class PandasParamHistoryManager(ParamHistoryManagerBase):
    """The parameter history manager based on ``pandas.DataFrame``."""
    def __init__(self, df: DataFrame):
        """
        Parameters:
            df: the data frame containing the previous records of the parameters and the evaluation results.
        """
        self.df = df

    def _df_has_value_set(self, df: DataFrame, values: Dict[str, Any]) -> bool:
        """Whether the df has a set of values.

        Parameters:
            df: the data frame representing the previous run results.
            values: the dictionary representing the key-value pairs of a single record.

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

    def filter(self, param_grid: List[Dict[str, Any]]) -> Iterable[Dict]:
        """
        Parameters:
            param_grid: the list of records to be filtered in.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame.from_dict([{'a': 11, 'b': 12, 'c': 10}])
        >>> param_grid = [{'a': 11, 'b': 12, 'c': 10}, {'a': 21, 'b': 22, 'c': 10}]
        >>> PandasParamHistoryManager(df).filter(param_grid)
        [{'a': 21, 'b': 22, 'c': 10}]

        This method considers the parameter to be existent even if the candidate contains only a subset of the values
        (i.e., implicitly assumes that no removal of the keys occurs along the development of the method).

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
