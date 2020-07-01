import pandas as pd
from typing import Optional


class AggregatorBase:
    pass


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
