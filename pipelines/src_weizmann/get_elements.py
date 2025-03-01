import torch
from svetlanna import SimulationParameters
from svetlanna.parameters import ConstrainedParameter
from svetlanna import elements


def get_free_space(
    freespace_sim_params,
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


def get_const_phase_layer(
    sim_params: SimulationParameters,
    value, max_phase=2 * torch.pi
):
    """
    Returns DiffractiveLayer with a constant phase mask.
    """
    x_nodes, y_nodes = sim_params.axes_size(axs=('W', 'H'))

    const_mask = torch.ones(size=(y_nodes, x_nodes)) * value
    
    return elements.DiffractiveLayer(
        simulation_parameters=sim_params,
        mask=ConstrainedParameter(
            const_mask,
            min_value=0,
            max_value=max_phase
        ),  # HERE WE ARE USING CONSTRAINED PARAMETER!
    )  # ATTENTION TO DOCUMENTATION!


def get_nonlinear_layer(
    sim_params: SimulationParameters,
):
    """
    Returns NonlinearElement ~ x^2.
        Comment: attention to the NonlinearElement documentation!
    """
    
    return elements.NonlinearElement(
        simulation_parameters=sim_params,
        response_function=lambda x: x ** 2,
    )
