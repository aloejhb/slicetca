import torch
from typing import Sequence, List

def get_component(components: Sequence[Sequence[torch.Tensor]], partition: int, product_pos:int):
    """
    Get component of a partition. Note that this will return the component for all ranks.

    :param components: components.
    :param partition: The index of the partition.
    :param product_pos: The index of the component within the product.

    :return: The component.

    Example:
        # For sliceTCA
        # X = \sum_{r=1}^R_1 u^r \times A^r + \sum_{r=1}^R_2 v^r \times B^r
        #     + \sum_{r=1}^R_2 w^r \times C^r
        # get the 1-th partition, 0-th component. This correspond to 
        get_component(components, 1, 0), which is v = (v^r)_{r=1}^{R_2}
    """
    return components[partition][product_pos]


def get_partition_weights(components: Sequence[Sequence[torch.Tensor]], partitions: List[int]):
    """
    Get the weight components for a list of partitions.

    :param components: components.
    :param partitions: The indices of the partitions.

    :return: The weights, which correspond to u, v or w, depending on the chosen partition.
    """
    for partition in partitions:
        yield components[partition][0]