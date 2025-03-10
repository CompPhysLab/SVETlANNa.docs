import torch
from svetlanna import SimulationParameters
from svetlanna.parameters import BoundedParameter
from svetlanna import Wavefront
from svetlanna import elements
from svetlanna.setup import LinearOpticalSetup


def get_const_free_space(
    freespace_sim_params: SimulationParameters,
    freespace_distance,  # in [m]!
    freespace_method='AS',
):
    """
    Returns FreeSpace layer with a bounded distance parameter.
    """
    return elements.FreeSpace(
        simulation_parameters=freespace_sim_params,
        distance=freespace_distance,  # distance is not learnable!
        method=freespace_method
    )


def get_random_diffractive_layer(
    difflayer_sim_params: SimulationParameters,
    mask_seed,  # for reproducability
    max_phase=torch.pi
):
    """
    Returns DiffractiveLayer with a random (or constant if mask_seed is not int) mask.
    """
    x_nodes, y_nodes = difflayer_sim_params.axes_size(axs=('W', 'H'))

    if isinstance(mask_seed, int):
        random_mask = torch.rand(
            size=(y_nodes, x_nodes),
            generator=torch.Generator().manual_seed(mask_seed)
        ) * (max_phase)
    else:
        random_mask = torch.ones(size=(y_nodes, x_nodes)) * mask_seed
    
    return elements.DiffractiveLayer(
        simulation_parameters=difflayer_sim_params,
        mask=BoundedParameter(
            random_mask,
            min_value=0,
            max_value=max_phase
        ),  # HERE WE ARE USING BOUNDED PARAMETER!
        mask_norm=1
    )

