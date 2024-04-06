from ..invariance.criteria import l2

def coupling_l2(x_tensors, y_tensors):
    """
    Compute the mean L2 norm between two lists of tensors.

    :param x_tensors: List of tensors.
    :param y_tensors: List of tensors.
    :return: Torch float.
    """
    diff = [x - y for x, y in zip(x_tensors, y_tensors)]
    return l2(diff)