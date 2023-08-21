import numpy as np
from copy import deepcopy

def data_sample(data_row, num_samples, rng=None, scaler = None, binary_data = False):
    """Generates a neighborhood around a prediction.


    Args:
        data_row: 1d numpy array, corresponding to a row
        num_samples: size of the neighborhood to learn the linear model
        rng: a numpy random number generator
        scaler: a scaler object
        binary_data: whether the data is binary (0/1) or continuous real data

    Returns:
        samples
    """
    # data = np.zeros((num_samples, data_row.shape[0]))

    if rng == None:
        rng = np.random.RandomState(12345)

    if not binary_data:
        tmp = rng.normal(0, 0.01, num_samples * data_row.shape[0]).astype(np.float32)
        epsilons = tmp.reshape(num_samples, data_row.shape[0])

        # broadcast
        # print isinstance(data_row, np.float32)
        # print isinstance(epsilons, np.float32)
        if scaler == None:
            data = (epsilons + data_row)
        else:
             data = (epsilons + data_row) * scaler.scale_ + scaler.mean_


    else:
        flip = rng.binomial(1, p, (num_samples, len(data_row))).astype(np.float32)
        orig = np.repeat(data_row[np.newaxis,:], num_samples, axis = 0)
        data = np.logical_xor(orig, flip).astype(np.float32)

    # the original data point
    data[0] = deepcopy(data_row)
    samples = deepcopy(data)

    return samples
