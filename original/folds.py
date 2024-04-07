import numpy as np
import numpy.typing as npt

def concat_all_folds(orig: npt.NDArray, except_fold: int) -> npt.NDArray:
    assert(except_fold) >= 0
    assert len(orig.shape) > 0
    # how do initialize our concatenated array:
    # 
    # - if except_fold is 0, initialize with index 1
    # - if except_fold is 1, initialize with index 0
    # 
    # in either case, we want to start at index 2 since we skip one of the 
    # previous indices (0 and 1) and choose to start with the other.
    concat = orig[1] if except_fold == 0 else orig[0]
    for i in range(2, orig.shape[0]):
        if i == except_fold:
            continue
        concat = np.concatenate((concat, orig[i]))
    return concat
