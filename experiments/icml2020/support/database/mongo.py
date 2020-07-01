import pandas as pd
import numpy as np
from pymongo import MongoClient
from sacred.observers import MongoObserver
from bson.objectid import ObjectId


def get_mongo_observer(MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASS,
                       MONGO_DBNAME):
    return MongoObserver(url=_get_mongo_uri(MONGO_HOST, MONGO_PORT, MONGO_USER,
                                            MONGO_PASS, MONGO_DBNAME))


def get_table(table_name, host, port, user, password, dbname):
    return _connect_mongo(host, port, user, password, dbname)[table_name]


def record_exists(collection, conditions):
    return collection.find(conditions).limit(1).count() > 0


def get_df(collection, query, db, host, port, username, password, no_id):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host,
                        port=port,
                        username=username,
                        password=password,
                        db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']
    return df


def get_row(collection, ids, db, host, port, username, password, no_id):
    return get_df(collection,
                  query={
                      '_id': ObjectId(ids)
                  },
                  db=db,
                  host=host,
                  port=port,
                  username=username,
                  password=password,
                  no_id=no_id).loc[0]


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = _get_mongo_uri(host, port, username, password, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def _get_mongo_uri(host, port, username, password, db):
    if username == '':
        return 'mongodb://%s:%s/%s' % (host, port, db)
    elif password == '':
        return 'mongodb://%s@%s:%s/%s' % (username, host, port, db)
    else:
        return 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port,
                                             db)
