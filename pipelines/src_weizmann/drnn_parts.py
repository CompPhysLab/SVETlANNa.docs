import torch
from .get_elements import *
from svetlanna.detector import Detector
from svetlanna import elements


def get_readin_layer(sim_params, fs_method):
    """
    Returns nn.Sequential for a Read-in layer of an optical RNN.
    """
    elements_list = []

    # FREE SPACE
    if VARIABLES['read_in_freespace']:
        elements_list.append(
            get_free_space(
                sim_params,
                DISTANCE,  # in [m]!
                freespace_method=fs_method,
            )
        )
    
    if APERTURES:  # APERTURE if needed
        elements_list.append(
            elements.Aperture(
                sim_params,
                mask=APERTURE_MASK,
            )
        )

    # PHASE LAYER (Diffractive layer)
    elements_list.append(
        get_const_phase_layer(
            sim_params,
            INIT_PHASE, max_phase=MAX_PHASE
        )
    )
    
    if READ_IN_NONLIN:  # NONLINEAR LAYER if needed
        elements_list.append(
            get_nonlinear_layer(sim_params),  # <- nonlinear layer after DiffractiveLayer!
        )

    # FREE SPACE
    elements_list.append(
        get_free_space(
            sim_params,
            DISTANCE,  # in [m]!
            freespace_method=fs_method,
        )
    )
    
    return nn.Sequential(*elements_list)


def get_memory_layer(sim_params, fs_method):
    """
    Returns nn.Sequential for a Memory layer of an optical RNN.
    """
    elements_list = []
    
    if APERTURES:  # APERTURE if needed
        elements_list.append(
            elements.Aperture(
                sim_params,
                mask=APERTURE_MASK,
            )
        )

    # PHASE LAYER (Diffractive layer)
    elements_list.append(
        get_const_phase_layer(
            sim_params,
            INIT_PHASE, max_phase=MAX_PHASE
        )
    )
        
    if MEMORY_NONLIN:  # NONLINEAR LAYER if needed
        elements_list.append(
            get_nonlinear_layer(sim_params),  # <- nonlinear layer after DiffractiveLayer!
        )

    # FREE SPACE
    elements_list.append(
        get_free_space(
            sim_params,
            DISTANCE,  # in [m]!
            freespace_method=fs_method,
        )
    )

    return nn.Sequential(*elements_list)


def get_readout_layer(sim_params, fs_method):
    """
    Returns nn.Sequential for a Read-out layer of an optical RNN.
    """
    elements_list = []

    if APERTURES:  # APERTURE if needed
        elements_list.append(
            elements.Aperture(
                sim_params,
                mask=APERTURE_MASK,
            )
        )

    # PHASE LAYER (Diffractive layer)
    elements_list.append(
        get_const_phase_layer(
            sim_params,
            INIT_PHASE, max_phase=MAX_PHASE
        )
    )
        
    if READ_OUT_NONLIN:  # NONLINEAR LAYER if needed
        elements_list.append(
            get_nonlinear_layer(sim_params),  # <- nonlinear layer after DiffractiveLayer!
        )

    # FREE SPACE
    elements_list.append(
        get_free_space(
            sim_params,
            DISTANCE,  # in [m]!
            freespace_method=fs_method,
        )
    )

    return nn.Sequential(*elements_list)
