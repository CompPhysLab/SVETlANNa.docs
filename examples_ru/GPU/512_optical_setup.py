import os
import sys
import random
import torch
import time
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
# our library
from svetlanna import SimulationParameters
from svetlanna.parameters import ConstrainedParameter
from svetlanna import Wavefront
from svetlanna import elements
from svetlanna.setup import LinearOpticalSetup
from svetlanna.detector import Detector, DetectorProcessorClf

from svetlanna.transforms import ToWavefront

# datasets of wavefronts
from src.wf_datasets import DatasetOfWavefronts
from src.wf_datasets import WavefrontsDatasetSimple

from tqdm import tqdm

from datetime import datetime

working_frequency = 0.4 * 1e12 # [Hz]
c_const = 299_792_458  # [m / s]

working_wavelength = c_const / working_frequency  # [m]

neuron_size = 0.53 * working_wavelength  # [m]

DETECTOR_SIZE = (128, 128)  # from the extended article!
# number of neurons in simulation
x_layer_nodes = DETECTOR_SIZE[1] * 1
y_layer_nodes = DETECTOR_SIZE[0] * 1
# Comment: Same size as proposed!


# physical size of each layer (from the article) - (8 x 8) [cm]
x_layer_size_m = x_layer_nodes * neuron_size  # [m]
y_layer_size_m = y_layer_nodes * neuron_size
# simulation parameters for the rest of the notebook
device = torch.device("cuda")
print(torch.cuda.is_available())
print(device)
SIM_PARAMS = SimulationParameters(
    axes={
        'W': torch.linspace(-x_layer_size_m / 2, x_layer_size_m / 2, x_layer_nodes),
        'H': torch.linspace(-y_layer_size_m / 2, y_layer_size_m / 2, y_layer_nodes),
        'wavelength': working_wavelength,  # only one wavelength!
    }
)

MNIST_DATA_FOLDER = './data'  # folder to store data
# TRAIN (images)
mnist_train_ds = torchvision.datasets.MNIST(
    root=MNIST_DATA_FOLDER,
    train=True,  # for train dataset
    download=False,
)
# TEST (images)
mnist_test_ds = torchvision.datasets.MNIST(
    root=MNIST_DATA_FOLDER,
    train=False,  # for test dataset
    download=False,
)

import src.detector_segmentation as detector_segmentation
number_of_classes = 10


detector_segment_size = 6.8 * working_wavelength

# size of each segment in neurons
x_segment_nodes = int(detector_segment_size / neuron_size)
y_segment_nodes = int(detector_segment_size / neuron_size)
# each segment of size = (y_segment_nodes, x_segment_nodes)


y_boundary_nodes = y_segment_nodes * 9
x_boundary_nodes = x_segment_nodes * 9


DETECTOR_MASK = detector_segmentation.squares_mnist(
    y_boundary_nodes, x_boundary_nodes,  # size of a detector or an aperture (in the middle of detector)
    SIM_PARAMS
)

ZONES_HIGHLIGHT_COLOR = 'r'
ZONES_LW = 0.5
selected_detector_mask = DETECTOR_MASK.clone().detach()


def get_zones_patches(detector_mask):
    """
    Returns a list of patches to draw zones in final visualisation
    """
    zones_patches = []

    delta = 1 #0.5

    for ind_class in range(number_of_classes):
        idx_y, idx_x = (detector_mask == ind_class).nonzero(as_tuple=True)

        zone_rect = patches.Rectangle(
            (idx_x[0] - delta, idx_y[0] - delta),
            idx_x[-1] - idx_x[0] + 2 * delta, idx_y[-1] - idx_y[0] + 2 * delta,
            linewidth=ZONES_LW,
            edgecolor=ZONES_HIGHLIGHT_COLOR,
            facecolor='none'
        )

        zones_patches.append(zone_rect)

    return zones_patches

# select modulation type
MODULATION_TYPE = 'amp'  # using ONLY amplitude to encode each picture in a Wavefront!

resize_y = int(DETECTOR_SIZE[0] / 3)
resize_x = int(DETECTOR_SIZE[1] / 3)  # shape for transforms.Resize
# Comment: Looks like in [2] article MNIST pictures were resized to ~100 x 100 neurons

# paddings along OY
pad_top = int((y_layer_nodes - resize_y) / 2)
pad_bottom = y_layer_nodes - pad_top - resize_y
# paddings along OX
pad_left = int((x_layer_nodes - resize_x) / 2)
pad_right = x_layer_nodes - pad_left - resize_x  # params for transforms.Pad


# compose all transforms!
image_transform_for_ds = transforms.Compose(
  [
      transforms.ToTensor(),
      transforms.Resize(
          size=(resize_y, resize_x),
          interpolation=InterpolationMode.NEAREST,
      ),
      transforms.Pad(
          padding=(
              pad_left,  # left padding
              pad_top,  # top padding
              pad_right,  # right padding
              pad_bottom  # bottom padding
          ),
          fill=0,
      ),  # padding to match sizes!
      ToWavefront(modulation_type=MODULATION_TYPE)  # <- select modulation type!!!
  ]
)

# TRAIN dataset of WAVEFRONTS
mnist_wf_train_ds = DatasetOfWavefronts(
    init_ds=mnist_train_ds,  # dataset of images
    transformations=image_transform_for_ds,  # image transformation
    sim_params=SIM_PARAMS,  # simulation parameters
    target='detector',
    detector_mask=DETECTOR_MASK
)


# TEST dataset of WAVEFRONTS
mnist_wf_test_ds = DatasetOfWavefronts(
    init_ds=mnist_test_ds,  # dataset of images
    transformations=image_transform_for_ds,  # image transformation
    sim_params=SIM_PARAMS,  # simulation parameters
    target='detector',
    detector_mask=DETECTOR_MASK
)

# plot several EXAMPLES from TRAIN dataset
n_examples= 4  # number of examples to plot
# choosing indecies of images (from train) to plot
random.seed(78)
train_examples_ids = random.sample(range(len(mnist_train_ds)), n_examples)

all_examples_wavefronts = []

n_lines = 3


NUM_OF_DIFF_LAYERS = 5  # number of diffractive layers
FREE_SPACE_DISTANCE = 40 * working_wavelength  # [m] - distance between difractive layers

MAX_PHASE = 2 * np.pi  # max phase for phase masks


FREESPACE_METHOD = 'AS'  # we use another method in contrast to [2]!!!


INIT_PHASES = torch.ones(NUM_OF_DIFF_LAYERS) * np.pi  # initial values for phase masks


# In[ ]:





# **<span style="color:red">Comment</span>**
#
# Here we are using a default `ConstrainedParameter` which is using the sigmoid function to limit a parameter range.
#
# In [[2]](https://ieeexplore.ieee.org/abstract/document/8732486) authors discuss such approach in the section Results and Discussion A. and underline that limiting parameters with the sigmoid function may lead to Vanishing Gradients. Authors also propose an another way to limit parameters - by using ReLU.
#
# In our case the sigmoid function works well but it is possible to realize the ReLU approach via specifying `bound_func` for `Constrained Parameter` (<span style="color:red">examples of customizing `bound_func` are provided in ...</span>).

# In[48]:


# functions that return single elements for further architecture

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


# In[49]:


def get_const_phase_layer_no_train(
    sim_params: SimulationParameters,
    value
):
    """
    Returns DiffractiveLayer with a constant phase mask.
    """
    x_nodes, y_nodes = sim_params.axes_size(axs=('W', 'H'))

    const_mask = torch.ones(size=(y_nodes, x_nodes)) * value

    return elements.DiffractiveLayer(
        simulation_parameters=sim_params,
        mask=const_mask,  # HERE WE ARE USING CONSTRAINED PARAMETER!
    )  # ATTENTION TO DOCUMENTATION!


# Function to construct a list of elements to reproduce an architecture from [the extended article](https://ieeexplore.ieee.org/abstract/document/8732486):

# In[50]:


def get_elements_list(
    num_layers,
    num_layers_no_train,
    simulation_parameters,
    freespace_method,
    phase_values,
):
    """
    Composes a list of elements for the setup.
    ...

    Parameters
    ----------
    num_layers : int
        Number of layers in the system.
    simulation_parameters : SimulationParameters()
        A simulation parameters for a task.
    freespace_method : str
        Propagation method for free spaces in a setup.
    phase_values : torch.Tensor()
        Torch tensor of phase values to generate constant masks for diffractive layers.

    Returns
    -------
    elements_list : list(Element)
        List of Elements for an optical setup.
    """
    elements_list = []  # list of elements

    # first FreeSpace layer before first DiffractiveLayer
    elements_list.append(
        get_free_space(
            simulation_parameters,  # simulation parameters for the notebook
            FREE_SPACE_DISTANCE,  # in [m]
            freespace_method=freespace_method,
        )
    )

    # compose the architecture
    for ind_layer in range(num_layers):

        # -----------------------------------------------------------------------
        # add DiffractiveLayer (learnable phase mask)
        elements_list.append(
            get_const_phase_layer(
                simulation_parameters,  # simulation parameters for the notebook
                value=phase_values[ind_layer].item(),
                max_phase=MAX_PHASE
            )
        )
        # -----------------------------------------------------------------------

        # add FreeSpace
        elements_list.append(
            get_free_space(
                simulation_parameters,  # simulation parameters for the notebook
                FREE_SPACE_DISTANCE,  # in [m]
                freespace_method=freespace_method,
            )
        )
    # print(1111111111111111111111)
    for ind_layer in range(num_layers_no_train):

        # -----------------------------------------------------------------------
        # add DiffractiveLayer (untrained phase mask)
        elements_list.append(
            get_const_phase_layer_no_train(
                simulation_parameters,  # simulation parameters for the notebook
                value=phase_values[(ind_layer % (num_layers))].item()
            )
        )
        # -----------------------------------------------------------------------

        # add FreeSpace
        elements_list.append(
            get_free_space(
                simulation_parameters,  # simulation parameters for the notebook
                FREE_SPACE_DISTANCE,  # in [m]
                freespace_method=freespace_method,
            )
        )

    # ---------------------------------------------------------------------------
    # add Detector in the end of the system!
    elements_list.append(
        Detector(
            simulation_parameters=simulation_parameters,
            func='intensity'  # detector that returns intensity
        )
    )

    return elements_list


# In[51]:


NUM_OF_DIFF_LAYERS_NO_TRAIN = 507


# In[52]:


architecture_elements_list = get_elements_list(
    num_layers=NUM_OF_DIFF_LAYERS,
    num_layers_no_train=NUM_OF_DIFF_LAYERS_NO_TRAIN,
    simulation_parameters=SIM_PARAMS,
    freespace_method=FREESPACE_METHOD,
    phase_values=INIT_PHASES,
)

print(f'Number of elements in the system (including Detector): {len(architecture_elements_list)}')


# In[ ]:





# ### 3.1.2. Compose `LinearOpticalSetup`

# In[53]:


def get_setup(simulation_parameters):
    """
    Returns an optical setup. Recreates all elements.
    """
    elements_list = get_elements_list(
        num_layers=NUM_OF_DIFF_LAYERS,
        num_layers_no_train=NUM_OF_DIFF_LAYERS_NO_TRAIN,
        simulation_parameters=SIM_PARAMS,
        freespace_method=FREESPACE_METHOD,
        phase_values=INIT_PHASES,
    )  # recreate a list of elements

    return LinearOpticalSetup(elements=elements_list)


# In[ ]:





# In[54]:


# creaye an optical setup
optical_setup = get_setup(SIM_PARAMS)


# In[55]:


optical_setup.net


# **<span style="color:red">Comment:</span>** Setup ends with `Detector` that returns an output tensor of intensities for each input `Wavefront`.

# In[ ]:





# #### Example of a wavefrnt propagation

# In[56]:


example_wf = mnist_wf_train_ds[128][0]


# In[57]:


setup_scheme, wavefronts = optical_setup.stepwise_forward(example_wf)


# In[ ]:





# ### 3.1.3 Detector processor (to calculate accuracies only)
#
# > ... size of these detectors $(6.4 \lambda \times 6.4 \lambda)$ ...
#
# **<span style="color:red">Comment:</span>** `DetectorProcessor` in our library is used to process an information on detector. For example, for the current task `DetectorProcessor` must return only 10 values (1 value per 1 class).

# In[58]:


CALCULATE_ACCURACIES = False  # if False, accuracies will not be calculated!


# In[59]:


# create a DetectorProcessorOzcanClf object
if CALCULATE_ACCURACIES:
    detector_processor = DetectorProcessorClf(
        simulation_parameters=SIM_PARAMS,
        num_classes=number_of_classes,
        segmented_detector=DETECTOR_MASK,
    )
else:
    detector_processor = None


# In[ ]:





# # 4. Training of the network
#
# Variables at the moment
# - `lin_optical_setup` : `LinearOpticalSetup` – a linear optical network composed of Elements
# - `detector_processor` : `DetectorProcessorClf` – this layer process an image from the detector and calculates probabilities of belonging to classes.

# In[60]:


DEVICE = 'cuda'  # 'mps' is not support a CrossEntropyLoss


# ## 4.1. Prepare some stuff for training

# ### 4.1.1. `DataLoader`'s
#
# Info from a supplementary material of [[1]](https://www.science.org/doi/suppl/10.1126/science.aat8084/suppl_file/aat8084-lin-sm-rev-3.pdf) for MNIST classification:
#
# > The training batch size was set to be $8$...

# In[61]:


train_bs = 128  # a batch size for training set
val_bs = 64


# > Forthis task, phase-only transmission masks weredesigned by training a five-layer $D^2 NN$ with $55000$ images ($5000$ validation images) from theMNIST (Modified National Institute of Stan-dards and Technology) handwritten digit data-base.

# In[62]:


# mnist_wf_train_ds
train_wf_ds, val_wf_ds = torch.utils.data.random_split(
    dataset=mnist_wf_train_ds,
    lengths=[55000, 5000],  # sizes from the article
    generator=torch.Generator().manual_seed(178)  # for reproducibility
)


# In[63]:


train_wf_loader = torch.utils.data.DataLoader(
    train_wf_ds,
    batch_size=train_bs,
    shuffle=True,
    # num_workers=2,
    drop_last=False,
)

val_wf_loader = torch.utils.data.DataLoader(
    val_wf_ds,
    batch_size=val_bs,
    shuffle=False,
    # num_workers=2,
    drop_last=False,
)


# In[ ]:





# ### 4.1.2. Optimizer and loss function
#
# Info from a supplementary material of [[1]](https://www.science.org/doi/suppl/10.1126/science.aat8084/suppl_file/aat8084-lin-sm-rev-3.pdf) for MNIST classification:
#
# > We used the stochastic gradient descent algorithm, Adam, to back-propagate the errors and update the
# layers of the network to minimize the loss function.
#
# **<span style="color:red">Additional info</span>** from [[2]](https://ieeexplore.ieee.org/abstract/document/8732486):
# > a back-propagation method by applying the adaptive moment estimation optimizer (Adam) with a learning rate of $10^{−3}$

# In[64]:


LR = 1e-3


# In[65]:


def get_adam_optimizer(net):
    return torch.optim.Adam(
        params=net.parameters(),  # NETWORK PARAMETERS!
        lr=LR
    )


# **<span style="color:red">Comment:</span>** We are using `MSELoss` as in [1] (that was clarified in [2])!

# In[66]:


loss_func_clf = nn.MSELoss()  # by default: reduction='mean'
loss_func_name = 'MSE'


# ## 4.2. Training of the optical network

# ### 4.2.1. Before training

# > a diffractive layer ... neurons ... were initialized with $\pi$ for phase values and $1$ for amplitude values ...

# #### Metrics for Test dataset

# In[67]:


test_wf_loader = torch.utils.data.DataLoader(
    mnist_wf_test_ds,
    batch_size=10,
    shuffle=False,
    # num_workers=2,
    drop_last=False,
)  # data loader for a test MNIST data


# In[68]:


optical_setup.net = optical_setup.net.to(DEVICE)
SIM_PARAMS = SIM_PARAMS.to(DEVICE)


# ## 4.3. Forward Loop

# In[ ]:


# def onn_forward(
#     optical_net,
#     wavefronts_dataloader,
#     device='cpu',
#     show_process=False
#     ):
#     """
#     Performs network's forward for all data.
#         NO DetectorProcessor
#         NO Loss
#     """
#     optical_net.eval()  # activate 'eval' mode of a model

#     for batch_wavefronts, batch_labels in tqdm(
#         wavefronts_dataloader,
#         total=len(wavefronts_dataloader),
#         desc='validation', position=0,
#         leave=True, disable=not show_process
#     ):  # go by batches
#         # batch_wavefronts - input wavefronts, batch_labels - labels
#         batch_size = batch_wavefronts.size()[0]

#         batch_wavefronts = batch_wavefronts.to(device, non_blocking=True)
#         batch_labels = batch_labels.to(device, non_blocking=True)

#         with torch.no_grad():
#             # ONLY FORWARD!
#             detector_output  = optical_net(batch_wavefronts)


# In[ ]:


import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

def onn_forward(
    optical_net,
    wavefronts_dataloader,
    device='cpu',
    show_process=False
    ):
    """
    Performs network's forward for all data.
        NO DetectorProcessor
        NO Loss
    """
    optical_net.eval()  # activate 'eval' mode of a model

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("onn_forward"):
            for batch_wavefronts, batch_labels in tqdm(
                wavefronts_dataloader,
                total=len(wavefronts_dataloader),
                desc='validation', position=0,
                leave=True, disable=not show_process
            ):  # go by batches
                # batch_wavefronts - input wavefronts, batch_labels - labels
                batch_size = batch_wavefronts.size()[0]

                batch_wavefronts = batch_wavefronts.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                with torch.no_grad():
                    # ONLY FORWARD!
                    detector_output = optical_net(batch_wavefronts)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


onn_forward(
        optical_setup.net,  # optical network composed in 3.
        train_wf_loader,  # dataloader of training set
        device=DEVICE,
        show_process=True,
    )
# In[ ]:


# from cProfile import Profile
# from pstats import SortKey, Stats


# ## 4.4. Profiling CPU and GPU

# In[ ]:


# # speed test just for forward on train set
# with Profile() as profile:
#     onn_forward(
#         optical_setup.net,  # optical network composed in 3.
#         train_wf_loader,  # dataloader of training set
#         device=DEVICE,
#         show_process=True,
#     )
#     Stats(profile).sort_stats(SortKey.CUMULATIVE).print_stats(30)


# In[ ]:


# params = [
#     {
#         'pin_memory': True,
#         'drop_last': True,
#     },
# ]

# cpu_times = {}
# gpu_times = {}
# batch_sizes = [2, 32, 128, 256, 512]

# for i, param in enumerate(params):
#     cpu_times_ = []
#     gpu_times_ = []

#     cpu_times[i] = cpu_times_
#     gpu_times[i] = gpu_times_

#     for bs in batch_sizes:
#         train_loader = torch.utils.data.DataLoader(
#             train_wf_ds,
#             batch_size=bs,
#             # shuffle=True,
#             # shuffle=False,
#             # num_workers=1,
#             # drop_last=False,
#             # pin_memory=True
#             **param

#         )

#         SIM_PARAMS = SIM_PARAMS.to('cpu')

#         optical_setup = get_setup(SIM_PARAMS)


#         net = optical_setup.net.to('cpu')
#         start = time.process_time()
#         onn_forward(
#             net,  # optical network composed in 3.
#             train_loader,  # dataloader of training set
#             device='cpu',
#             show_process=True,
#         )
#         cpu_times_.append(time.process_time() - start)


#         SIM_PARAMS = SIM_PARAMS.to('cuda')

#         optical_setup = get_setup(SIM_PARAMS)

#         net = optical_setup.net.to('cuda')
#         start = time.process_time()
#         onn_forward(
#             net,  # optical network composed in 3.
#             train_loader,  # dataloader of training set
#             device='cuda',
#             show_process=True,
#         )
#         gpu_times_.append(time.process_time() - start)


# In[ ]:


# plt.figure(figsize=(6, 3))
# plt.plot(batch_sizes, cpu_times[0], '-o', label='cpu')
# plt.plot(batch_sizes, gpu_times[0], '-o', label='gpu')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('bs')
# plt.ylabel('time, s')
# plt.legend()

# if __name__ == "__main__":
#     # Initialize your optical setup and dataloaders here
#     optical_setup = get_setup(SIM_PARAMS)
#     train_wf_loader = torch.utils.data.DataLoader(
#         train_wf_ds,
#         batch_size=128,
#         shuffle=True,
#         drop_last=False,
#     )

#     onn_forward(
#         optical_setup.net,  # optical network composed in 3.
#         train_wf_loader,  # dataloader of training set
#         device='cuda',
#         show_process=True,
#     )