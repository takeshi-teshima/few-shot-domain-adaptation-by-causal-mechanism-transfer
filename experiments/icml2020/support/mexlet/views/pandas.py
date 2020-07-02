import pandas as pd


def pd_add_column(df_a, df_b, index_key, index_key_b=None):
    if index_key_b is None:
        index_key_b = index_key
    df_b = df_b.rename(columns={index_key_b: index_key})
    df_a = pd.merge(df_a, df_b, on=index_key)
    return df_a
