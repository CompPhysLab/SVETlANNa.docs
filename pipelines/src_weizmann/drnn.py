import torch
from torch import nn
from svetlanna import SimulationParameters
from svetlanna import Wavefront, elements
from svetlanna.detector import Detector
from svetlanna.parameters import ConstrainedParameter
from .get_elements import *


class DiffractiveRNN(nn.Module):
    """
    A recurrent diffractive netvork proposed in [1].
    """

    def __init__(
        self,
        sequence_size: int,
        fusing_coeff: float,
        sim_params: SimulationParameters,
        fs_method: str = 'AS',
        device: str | torch.device = torch.get_default_device(),
    ):
        """
        sequence_size: int
            A selected size (number of frames) of sub-sequences for action prediction.
        fusing_coeff: float
            A coefficient in a function for a hidden state (lambda in methods [1]).
        sim_params: SimulationParameters
            Simulation parameters for the task.
        fs_method: str
            A method for a free space.
        """
        super().__init__()

        self.sequence_len = sequence_size
        self.fusing_coeff = fusing_coeff
        self.sim_params = sim_params

        self.h, self.w = self.sim_params.axes_size(
            axs=('H', 'W')
        )  # height and width for a wavefronts
        
        self.__device = device
        self.fs_method = fs_method
        
        self.free_space_optimal = get_free_space(
            self.sim_params,
            DISTANCE,  # in [m]!
            freespace_method=self.fs_method,
        ).to(self.__device)  # untrainable free space between timesteps

        # ----------------------------------------------- READ-IN LAYER
        self.read_in_layer = get_readin_layer(
            self.sim_params, self.fs_method,
        ).to(self.__device)
        # ------------------------------------------------ MEMORY LAYER
        self.memory_layer = get_memory_layer(
            self.sim_params, self.fs_method,
        ).to(self.__device)
        # ---------------------------------------------- READ-OUT LAYER
        self.read_out_layer = get_readout_layer(
            self.sim_params, self.fs_method,
        ).to(self.__device).to(self.__device)
        # -------------------------------------------------------------
        # -------------------------------------------- DETECTOR (LAYER)
        self.detector = get_detector_layer(self.sim_params).to(self.__device)
        # -------------------------------------------------------------

    def forward(self, subsequence_wf):
        """
        Parameters
        ----------
        subsequence_wf: Wavefront('time', 'H', 'W')
            List of wavefronts for a video sub-sequence.
        """
        if len(subsequence_wf.shape) > 3:  # if a batch is an input
            batch_flag = True
            bs = subsequence_wf.shape[0]
            h_prev = Wavefront(
                torch.zeros(
                    size=(bs, self.h, self.w)
                )
            ).to(self.__device) # h_{t - 1} - reset hidden for the first input
        else:
            batch_flag = False
            h_prev = Wavefront(
                torch.zeros(
                    size=(self.h, self.w)
                )
            ).to(self.__device) # h_{t - 1} - reset hidden for the first input

        for frame_ind in range(self.sequence_len):
            if batch_flag:
                x_t = subsequence_wf[:, frame_ind, :, :]
            else:  # not a batch
                x_t = subsequence_wf[frame_ind, :, :]
                
            i_t = self.read_in_layer(x_t)  # f_2(x_t)

            m_t = self.memory_layer(h_prev)  # f_1(h_{t - 1})
            
            h_prev = self.fusing_coeff * m_t + (1 - self.fusing_coeff) * i_t
            h_prev = self.free_space_optimal(h_prev)

        out = self.read_out_layer(h_prev)
        
        return self.detector(out)


################################################################################
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


def get_detector_layer(sim_params):
    if not APERTURES:
        return Detector(
            sim_params,
            func='intensity'  # detector that returns intensity
        )
    else:  # add aperture before detector if needed
        return nn.Sequential(
            elements.Aperture(
                sim_params,
                mask=APERTURE_MASK,
            ),  # <- aperture
            # get_nonlinear_layer(self.sim_params),  # <- nonlinear layer after DiffractiveLayer!
            Detector(
                sim_params,
                func='intensity'  # detector that returns intensity
            )
        )
