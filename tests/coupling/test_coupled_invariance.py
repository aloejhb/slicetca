import torch
from slicetca.core import SliceTCA
from slicetca.invariance.transformations import TransformationBetween
from slicetca.coupling.coupling_criteria import coupling_l2
from slicetca.coupling.coupled_iterative_invariance import sgd_coupled_invariance

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test_sdg_coupled_invariance():
    n_neuron_list = [10, 20, 35]
    n_time = 12
    n_trial = 3

    tca_params = {
        'ranks': (2, 0, 1),
        'device': device,
    }

    compos = [torch.randn(n_trial, n_neuron, n_time,
                          device=device)
                    for n_neuron in n_neuron_list]
    
    models = []
    for compo in compos:
        tca_params['dimensions'] = compo.shape
        models.append(SliceTCA(**tca_params))

    models = sgd_coupled_invariance(
        models=models,
        objective_function=coupling_l2,
        transformation_class=TransformationBetween,
        coupled_partitions=[0, 2],
        learning_rate=1e-2,
        max_iter=100,
        verbose=False,
        progress_bar=True
    )

    # Assert that some transformation has occurred (very basic check)
    for i, model in enumerate(models):
        assert not torch.allclose(initial_components[i], model.get_components(detach=True)), "Component should be transformed"