import numpy as np
import numpy.typing as npt

def _replace_final_dim(orig_x: npt.NDArray, final_dim_size: int) -> npt.NDArray:
    '''
    Given an NDArray `orig_x`, return a new NDArray of the same shape,
    except with final dimension `final_dim_size`.
    '''
    orig_x_shape = orig_x.shape
    x_transform = np.zeros(
        (*orig_x_shape[:-1], final_dim_size)
    )
    assert x_transform.shape[:-1] == orig_x.shape[:-1]
    return x_transform

def transform_for_channels(x: npt.NDArray, channels: list[int]) -> npt.NDArray:
    '''
    Returns an NDArray whose shape is identical to that of the `x` parameter,
    except the final dimension is of size `len(channels)`

    Generally speaking, x has a shape similar to 
    (NUM_FOLDS, 530, 1920, NUM_CHANNELS). In this array, each fold thus
    has shape (530, 1920, NUM_CHANNELS).
    
    Since we're trying to only extract `num_channels` out into the last 
    dimension, we need to create an NDArray whose final dimension matches
    that number of channels.
     
    The value returned from this function, which we'll call `x_transform`,
    is compatible with this number of channels since its last dimension is 
    the number of channels we're trying to extract.
    '''

    num_channels = len(channels)
    assert (len(x.shape)) == 4
    x_transform = _replace_final_dim(
        orig_x=x,
        final_dim_size=num_channels,
    )
    return x_transform
