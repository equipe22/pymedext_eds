import functools
import time
import re
from logzero import logger

from pymedextcore.document import Document
from pymedextcore.annotators import Attribute
from git import Repo,InvalidGitRepositoryError

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def chunks(lst, n_chunks):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n_chunks):
        yield lst[i:i + n_chunks]


def to_chunks(lst, n_chunks):
    """List of sublists of size n form lst
    :param lst: List
    :param n: Integer
    :returns: List"""
    res = []
    for i in range(0,len(lst), n_chunks):
        res.append(lst[i:i+n_chunks])
    return res

def rawtext_loader(file):
    """ Loads documents from text file
    :param file: path to a text file
    """
    with open(file) as f:
        txt = f.read()
        _id = re.search(r"([A-Za-z0-9]+)\.txt$", file)
        if not _id:
            _id = file
        else:
            _id  = _id.groups()[0]
    return Document(
        raw_text = txt,
        ID = _id,
        attributes = [Attribute(type='person_id', value=_id)]
    )

def get_version_git(annotator,
                    repo_name="equipe22/pymedext_eds"):
    """Get the git commit hash corresponding to the version of the annotator
    params: annotator: Name of the annotators
    params: repo_name: Name of the github repo
    """
    try:
        repo = Repo()
        commit = repo.commit('master').hexsha
        return f"{annotator}:{repo_name}:{commit}"
    except InvalidGitRepositoryError:
        return None
