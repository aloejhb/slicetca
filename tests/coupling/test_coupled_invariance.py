import torch
from slicetca.core import SliceTCA
from slicetca.invariance.transformations import TransformationBetween
from slicetca.coupling.coupling_criteria import coupling_l2
from slicetca.coupling.coupled_iterative_invariance import sgd_coupled_invariance
from slicetca.run.decompose import decompose

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test_sdg_coupled_invariance():
    n_neuron_list = [10, 20, 35]
    n_time = 12
    n_trial = 4

    compos = [torch.randn(n_trial, n_neuron, n_time,
                          device=device)
                    for n_neuron in n_neuron_list]
    
    models = []
    print('Fit SliceTCA')
    for compo in compos:
        _, model = decompose(compo,
                             number_components=(1, 2, 1),
                             max_iter=100
                            )
        models.append(model)

    initial_components = [model.get_components() for model in models]

    print('Coupled invariance')
    models = sgd_coupled_invariance(
        models=models,
        objective_function=coupling_l2,
        transformation_class=TransformationBetween,
        coupled_partitions=[1],
        learning_rate=1e-2,
        max_iter=100,
        verbose=False,
        progress_bar=True
    )

    # Assert that some transformation has occurred (very basic check)
    for i, model in enumerate(models):
        assert not torch.allclose(initial_components[i][1][1], model.get_components(detach=True)[1][1]), 'Component should be transformed'

if __name__ == '__main__':
    test_sdg_coupled_invariance()