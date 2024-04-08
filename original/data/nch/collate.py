from typing import Any
import numpy as np
import numpy.typing as npt
import tensorflow as tf

def max_dimensions(
        lst: list[Any],
        level: int=0,
        max_dims: list[int] | None = None
) -> tuple[int, ...]:
    """
    Finds the maximum dimension for each level of a nested list structure

    :param lst: The list for which to get dimensions
    :param level: INTERNAL USE ONLY (the dimension we are processing)
    :param max_dims: INTERNAL USE ONLY (the current array of maximums)
    :return: a tuple of sizes, similar to torch.Tensor.shape()
    """
    if max_dims is None:
        max_dims = []

    # Extend the max_dims list if this is the deepest level we've encountered so far
    if level >= len(max_dims):
        max_dims.append(len(lst))
    else:
        max_dims[level] = max(max_dims[level], len(lst))

    for item in lst:
        if isinstance(item, list):
            # Recursively process each sublist
            max_dimensions(item, level + 1, max_dims)

    return tuple(max_dims)

def pad_lists(
    lst: list[Any],
    pad_with: int = 0
) -> npt.NDArray:
    """
    Given a ragged nested list structure `lst` (i.e. where the length
    in each dimension is not uniform), return a new list with all
    levels of the list padded to the maximal length of any list in that 
    dimension. Padding elements will have the same value as given in 
    `pad_with`

    For example, the return value of `pad_lists([[1], [1, 2]], 0)` will
    be `[[1, 0], [1, 2]]`

    :param lst: the list to pad, if it is ragged. if it's not, this function
        is a no-op
    :param pad_with: the value to use for padding
    """



    # max_dims[0] is the number of elements (either lists or ints) we 
    # need in this dimension
    return tf.ragged.constant(lst).to_tensor(pad_with).numpy()
