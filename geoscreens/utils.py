import json
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def batchify(iterable, batch_size=1):
    """Splits an iterable / list-like into batches of size n"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


@contextmanager
def timeit_context(name):
    """
    Easy way to wrap your code and output how long it takes to run.

    Example::

        with utils.timeit_context("My profiling code"):
            do_something()
            do_more_things()

        Output:
        [My profiling code] finished in 0:00:04.321035

    """
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    time_delta = timedelta(seconds=elapsedTime)
    print(f"[{name}] finished in {time_delta}")


def load_json(json_path: Path) -> Dict[str, Any]:
    """
    Args:
        json_path: Path to json file

    Returns: json dictionary
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(
    json_path: Union[str, Path], data: Dict, indent: int = 4, sort_keys: bool = True
) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=indent,
            # cls=CustomJSONEncoder,
            sort_keys=sort_keys,
        )


def get_indices_to_sample(config, total_frames: int, fps: float) -> List[int]:
    indices = map(
        int,
        np.linspace(
            start=0.0,
            stop=total_frames,
            num=int(total_frames * (config.frame_sample_rate_fps / fps)),
            retstep=False,
            endpoint=False,
        ),
    )
    return list(indices)
