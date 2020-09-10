import pgpasslib
from sqlalchemy import create_engine
import pandas as pd


def get_engine(): 
    password = pgpasslib.getpass('10.172.28.101', 5432, 'coronaomop', 'coronascientist')
    return create_engine(f'postgresql+psycopg2://coronascientist:{password}@10.172.28.101:5432/coronaomop')

def construct_query(limit, min_date, view='unstable'): 
    if limit != -1:
        limit = "LIMIT {}".format(limit)
    else:
        limit = ""
        
    query = """
        SELECT note_id, person_id, note_text
        FROM {}.note 
        WHERE note_date > '{}'
        AND note_text IS NOT NULL OR note_text not in ('','\n') 
        {}
        """.format(view, min_date, limit)
    #logger.debug('query set with parameters: view = {}, min_date= {}, limit={}'.format(view, min_date, limit))
    return query


def rawtext_pg(limit, min_date,  engine,chunksize=100):
    query = construct_query(limit, min_date)
    return pd.read_sql_query(query, engine,chunksize=chunksize)