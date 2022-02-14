def batchify(iterable, batch_size=1):
    """Splits an iterable / list-like into batches of size n"""
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]
