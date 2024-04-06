import torch
import numpy as np
import tqdm
import copy
from typing import List, Callable
from ..core.decompositions import SliceTCA
from .helper import get_partition_weights


def sgd_coupled_invariance(models: List[SliceTCA],
                           objective_function: Callable,
                           transformation_class: Callable,
                           coupled_partitions: List[int],
                           learning_rate: float = 10**-2,
                           max_iter: int = 10000,
                           min_std: float = 10**-3,
                           iter_std: int = 100,
                           verbose: bool = False,
                           progress_bar: bool = True):
    """
    Optimizes transformations for a list of SliceTCA models to minimize the difference between their components.

    :param models: List of SliceTCA models.
    :param objective_function: Function to minimize differences between model components.
    :param transformation_class: Class to instantiate transformation objects.
    :param coupled_partitions: List of partitions to couple.
    :param learning_rate: Learning rate for the optimizer.
    :param max_iter: Maximum number of iterations.
    :param min_std: Minimum std of the last iter_std iterations under which to assume convergence.
    :param iter_std: Number of iterations to consider for convergence checking.
    :param verbose: Whether to print the loss.
    :param progress_bar: Whether to use a progress bar.
    """
    
    transformations = [transformation_class(model) for model in models]

    for model in models:
        model.requires_grad_(False)

    for transformation in transformations:
        transformation.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [param for transformation in transformations for param in transformation.parameters()],
        lr=learning_rate
    )

    losses = []

    iterator = tqdm.tqdm(range(max_iter)) if progress_bar else range(max_iter)

    for iteration in iterator:
        optimizer.zero_grad()

        total_loss = 0.0
        for i, model_i in enumerate(models):
            components_i = model_i.get_components(detach=True)
            for j, model_j in enumerate(models):
                if i == j:
                    continue
                components_j = model_j.get_components(detach=True)

                components_transformed_i = transformations[i](copy.deepcopy(components_i))
                components_transformed_j = transformations[j](copy.deepcopy(components_j))
                
                weights_i = get_partition_weights(components_transformed_i, coupled_partitions)
                weights_j = get_partition_weights(components_transformed_j, coupled_partitions)

                # Compute the objective function between the transformed components
                loss = objective_function(weights_i, weights_j)
                total_loss += loss

        if verbose: print(f'Iteration: {iteration}, Loss: {total_loss.item()}')
        if progress_bar: iterator.set_description(f'Coupled Invariance Loss: {total_loss.item():.4f} ')

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if len(losses) > iter_std and np.std(losses[-iter_std:]) < min_std:
            if progress_bar: iterator.set_description("Converged. Coupled Invariance Loss: {:.4f} ".format(losses[-1]))
            break

    # Apply the transformations to the models
    for i, model in enumerate(models):
        transformed_components = transformations[i](model.get_components(detach=True))
        model.set_components(transformed_components)

    return models
