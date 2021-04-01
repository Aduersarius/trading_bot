import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    #data = list(data["close"])
    block = data.iloc[t: t + n_days, :]   # pad with t0
    res = [0.5]
    for i in range(n_days-1):
        res.append(sigmoid(block["close"][i + 1] - block["close"][i]))
    block = block.to_numpy()
    block[:, 0] = res
    return block
