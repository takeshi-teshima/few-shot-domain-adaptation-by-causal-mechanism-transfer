import pandas as pd
from typing import Optional

class AggregatorBase:
    pass

class SQLAggregator(AggregatorBase):
    def __init__(self, session, Model, tag):
        self.session = session
        self.Model = Model
        self.tag = tag

    def get_results_pd(self):
        res = pd.DataFrame()
        query = self.session.query(
            self.Model).filter(self.Model.tag == self.tag)
        return pd.read_sql(query.statement, query.session.bind)


class MongoAggregator(AggregatorBase):
    def __init__(self, records, tag=None, query={}):
        self.records = records
        self.tag = tag
        self.query = query

    def get_results_pd(self, index: Optional[str] = '_id'):
        query = self.query.copy()
        if self.tag is not None:
            query.update({'tag': self.tag})
        df = pd.DataFrame(list(self.records.find(query)))
        if index is not None:
            del df[index]
        return df


class PdAggregator(AggregatorBase):
    def __init__(self, df):
        self.df = df

    def get_results_pd(self):
        return self.df
