
import functools
import time
from logzero import logger

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
    for i in range(0,len(lst), n):
        res.append(lst[i:i+n])
    return res