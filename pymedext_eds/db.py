import pgpasslib
from sqlalchemy import create_engine
import pandas as pd


def get_engine(): 
    password = pgpasslib.getpass('10.172.28.101', 5432, 'coronaomop', 'coronascientist')
    return create_engine(f'postgresql+psycopg2://coronascientist:{password}@10.172.28.101:5432/coronaomop')


def construct_query(limit, min_date='2020-03-01', view='unstable', note_ids = None): 
    if limit != -1:
        limit = "LIMIT {}".format(limit)
    else:
        limit = ""
        
    notes = ""
    if note_ids is not None:
        notes = f"AND note_id IN {tuple(note_ids)}"
        
    query = f"""
        SELECT note_id, person_id, note_text, note_date
        FROM {view}.note 
        WHERE note_date > '{min_date}'
        {notes}
        AND note_text IS NOT NULL OR note_text not in ('','\n', ' ') 
        {limit}
        """
    #logger.debug('query set with parameters: view = {}, min_date= {}, limit={}'.format(view, min_date, limit))
    return query

def rawtext_pg(limit, min_date,  engine,chunksize=100):
    query = construct_query(limit, min_date)
    return pd.read_sql_query(query, engine,chunksize=chunksize)