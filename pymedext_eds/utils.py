import functools
import re
import time

from git import Repo, InvalidGitRepositoryError
from logzero import logger

from typing import Optional, Union, Any
from pymedextcore.document import Document
from pymedextcore.annotators import Annotator
from pyspark.sql import types, functions as F, DataFrame, Column

SCHEMA = types.ArrayType(types.StructType([
    types.StructField("lexical_variant", types.StringType(), False),
    types.StructField("start", types.IntegerType(), False),
    types.StructField("end", types.IntegerType(), False),
    types.StructField("offset", types.StringType(), False),
    types.StructField("snippet", types.StringType(), False),
    types.StructField("term_modifiers", types.StringType(), False),
]))

def pymedext2omop(record):
    attr = record['attributes']

    lexical_variant = record['value']
    start, end = record['span']
    offset = f"{start},{end}"
    if 'snippet' in attr:
        snippet = attr['snippet']
    else:
        snippet = None

    modifiers = [
        f'{k}={v}'
        for k, v in attr.items()
        if k in {'hypothesis', 'context', 'negation', 'id_regex'}
    ]
    term_modifiers = ','.join(modifiers)

    return lexical_variant, start, end, offset, snippet, term_modifiers

def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def to_chunks(lst, n):
    """List of sublists of size n form lst
    :param lst: List
    :param n: Integer
    :returns: List"""
    res = []
    for i in range(0, len(lst), n):
        res.append(lst[i:i + n])
    return res


def rawtext_loader(file):
    with open(file) as f:
        txt = f.read()
        ID = re.search("([A-Za-z0-9]+)\.txt$", file)
        if not ID:
            ID = file
        else:
            ID = ID.groups()[0]
    return Document(
        raw_text=txt,
        ID=ID,
        attributes={'person_id': ID}
    )


def get_version_git(annotator,
                    repo_name="equipe22/pymedext_eds"):
    try:
        repo = Repo()
        commit = repo.commit('master').hexsha
        return f"{annotator}:{repo_name}:{commit}"
    except InvalidGitRepositoryError:
        return None
