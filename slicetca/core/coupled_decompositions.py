import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from slicetca.core import SliceTCA
from typing import List


class CoupledDecomposition(nn.Module):
    def __init__(self, composition_shapes, tca_params):
        """
        Initializes the model for decomposing multiple tensors with coupled constraints.
        """
        super().__init__()
        self.num_compositions = len(composition_shapes)
        self.decomposers = nn.ModuleList()
        for comp_shape in composition_shapes:
            tca_params['dimensions'] = comp_shape
            self.decomposers.append(SliceTCA(**tca_params))


    def construct_single_composition(self, index):
        """
        Constructs a single tensor from the decompositions.
        """
        X_hat = self.decomposers[index].construct()
        return X_hat


    def construct(self):
        """
        Constructs the matrices from the decompositions.
        """
        X_hats = [decomp.construct() for decomp in self.decomposers]
        return X_hats
    

    def total_loss(self, Xs, similarity_idxs, gamma=1.0):
        """
        Computes the total loss including reconstruction and similarity terms.

        Args:
        - alpha: Weighting parameter for the reconstruction losses.
        - gamma: Weighting parameter for the similarity losses.

        Returns:
        - Total loss value.
        """
        X_hats = self.construct()

        reconstruct_losses = [torch.norm(X - X_hat, p='fro')**2 / X.numel()
                              for X, X_hat in zip(Xs, X_hats)]
        reconstruction_loss = sum(reconstruct_losses) / len(reconstruct_losses)

        similarity_losses = []
        for similarity_idx in similarity_idxs:
            for n in range(self.num_compositions):
                for m in range(n + 1, self.num_compositions):
                    diff = self.decomposers[n].get_component(similarity_idx, 0) - self.decomposers[m].get_component(similarity_idx, 0)
                    similarity_losses.append(torch.norm(diff, p='fro')**2 / diff.numel())

        similarity_loss = sum(similarity_losses) / len(similarity_losses)
        loss = reconstruction_loss + gamma * similarity_loss
        return loss


    def fit(self,
            Xs: List[torch.Tensor],
            optimizer: torch.optim.Optimizer,
            similarity_idxs: List[int],
            similarity_gamma: float = 1.0,
            max_iter: int = 1000,
            min_std: float = 10 ** -3,
            iter_std: int = 100,
            verbose: bool = False,
            progress_bar: bool =True):
        """
        Fits the model to the data.
        """
        losses = []
        self.losses = []

        iterator = tqdm(range(max_iter)) if progress_bar else range(max_iter)

        for iteration in iterator:
            total_loss = self.total_loss(Xs, similarity_idxs, gamma=similarity_gamma)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

            if verbose: print('Iteration:', iteration, 'Loss:', total_loss)
            if progress_bar: iterator.set_description('Loss: ' + str(total_loss) + ' ')

            if len(losses) > iter_std and np.array(losses[-iter_std:]).std() < min_std:
                if progress_bar: iterator.set_description('The model converged. Loss: ' + str(total_loss) + ' ')
                break

        self.losses += losses


if __name__ == '__main__':
    # Example usage
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    num_compositions = 3
    n_neuron_list = [10, 20, 35]
    n_time = 12
    n_trial = 3

    tca_params = {
        'ranks': (0, 2, 1),
        'device': device,
    }

    compositions = [torch.randn(n_neuron, n_time, n_trial,
                                device=device)
                    for n_neuron in range(num_compositions)]

    model = CoupledDecomposition(composition_shapes=[comp.shape for comp in compositions], tca_params=tca_params)
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.fit(compositions, optimizer, similarity_idx=1, max_iter=50000, progress_bar=True)
