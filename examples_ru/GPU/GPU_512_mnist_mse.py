import os
import sys
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
# our library
from svetlanna import SimulationParameters
from svetlanna.parameters import ConstrainedParameter
# our library
from svetlanna import Wavefront
from svetlanna import elements
from svetlanna.setup import LinearOpticalSetup
from svetlanna.detector import Detector, DetectorProcessorClf
from svetlanna.transforms import ToWavefront
from svetlanna.clerk import Clerk
# datasets of wavefronts
from src.wf_datasets import DatasetOfWavefronts
from src.wf_datasets import WavefrontsDatasetSimple
import src.detector_segmentation as detector_segmentation
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('dark_background')

import gc
import pickle

# Ensure memory_snapshots directory exists
MEMORY_SNAPSHOTS_FOLDER = 'memory_snapshots'
if not os.path.exists(MEMORY_SNAPSHOTS_FOLDER):
    os.makedirs(MEMORY_SNAPSHOTS_FOLDER)

working_frequency = 0.4 * 1e12  # [Hz]
c_const = 299_792_458  # [m / s]
working_wavelength = c_const / working_frequency  # [m]

# neuron size (square)
neuron_size = 0.53 * working_wavelength  # [m]

DETECTOR_SIZE = (64, 64)
# an actual zone where weights will be updated during a training process

# number of neurons in simulation
x_layer_nodes = DETECTOR_SIZE[1] * 1
y_layer_nodes = DETECTOR_SIZE[0] * 1
# Comment: Same size as proposed!

# physical size of each layer [cm]
x_layer_size_m = x_layer_nodes * neuron_size  # [m]
y_layer_size_m = y_layer_nodes * neuron_size  # [m]

# simulation parameters for the rest of the notebook
SIM_PARAMS = SimulationParameters(
    axes={
        'W': torch.linspace(-x_layer_size_m / 2, x_layer_size_m / 2, x_layer_nodes),    # noqa: E501
        'H': torch.linspace(-y_layer_size_m / 2, y_layer_size_m / 2, y_layer_nodes),    # noqa: E501
        'wavelength': working_wavelength,  # only one wavelength!
    }
)

# initialize a directory for a dataset
MNIST_DATA_FOLDER = './data'  # folder to store data

# TRAIN (images)
mnist_train_ds = torchvision.datasets.MNIST(
    root=MNIST_DATA_FOLDER,
    train=True,  # for train dataset
    download=True,
)

# TEST (images)
mnist_test_ds = torchvision.datasets.MNIST(
    root=MNIST_DATA_FOLDER,
    train=False,  # for test dataset
    download=True,
)

number_of_classes = 10

# для сетки 1024x1024 было 22
detector_segment_size = 6 * working_wavelength

# size of each segment in neurons
x_segment_nodes = int(detector_segment_size / neuron_size)
y_segment_nodes = int(detector_segment_size / neuron_size)
# each segment of size = (y_segment_nodes, x_segment_nodes)

y_boundary_nodes = y_segment_nodes * 9
x_boundary_nodes = x_segment_nodes * 9


# This mask will be used to generate a target image for each number
DETECTOR_MASK = detector_segmentation.squares_mnist(
    y_boundary_nodes, x_boundary_nodes,  # size of a detector or an aperture
    # (in the middle of detector)
    SIM_PARAMS
)
# Target image: zeros are everywhere except the necessary zone responsible for
# the label!

# select modulation type
MODULATION_TYPE = 'amp'  # using ONLY amplitude to encode each picture in a
# Wavefront!

resize_y = int(DETECTOR_SIZE[0] / 3)
resize_x = int(DETECTOR_SIZE[1] / 3)  # shape for transforms.Resize

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
      ToWavefront(modulation_type=MODULATION_TYPE)  # <- select modulation
      # type!!!
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

NUM_OF_DIFF_LAYERS = 5  # number of diffractive layers that will be trained
FREE_SPACE_DISTANCE = 40 * working_wavelength  # [m] - distance between
# difractive layers

MAX_PHASE = 2 * torch.pi  # max phase for phase masks

FREESPACE_METHOD = 'AS'  # we use another method in contrast to [2]!!!

INIT_PHASE = torch.pi  # initial values for phase masks


def set_setup(
    total_number_of_layers: int,
    number_of_layers_at_the_beginning: int,
):

    global FREE_SPACE_DISTANCE, MAX_PHASE, FREESPACE_METHOD, INIT_PHASE
    global SIM_PARAMS

    elements_list = []

    free_space = elements.FreeSpace(
        simulation_parameters=SIM_PARAMS,
        distance=FREE_SPACE_DISTANCE,
        method=FREESPACE_METHOD
    )

    x_nodes, y_nodes = SIM_PARAMS.axes_size(axs=('W', 'H'))
    const_mask = torch.ones(size=(y_nodes, x_nodes)) * INIT_PHASE

    trainable_diffractive_layer = elements.DiffractiveLayer(
        simulation_parameters=SIM_PARAMS,
        mask=ConstrainedParameter(
            const_mask,
            min_value=0,
            max_value=MAX_PHASE
        ),
    )

    untrained_diffractive_layer = elements.DiffractiveLayer(
        simulation_parameters=SIM_PARAMS,
        mask=const_mask,  # HERE WE ARE DON'T USE CONSTRAINED PARAMETER!
    )

    elements_list.append(free_space)

    # TODO: add 2 layers
    for _ in range(1):
        elements_list.append(trainable_diffractive_layer)
        elements_list.append(free_space)

    for _ in range(number_of_layers_at_the_beginning):
        elements_list.append(untrained_diffractive_layer)
        elements_list.append(free_space)

    elements_list.append(trainable_diffractive_layer)
    elements_list.append(free_space)

    for _ in range(total_number_of_layers - number_of_layers_at_the_beginning):
        elements_list.append(untrained_diffractive_layer)
        elements_list.append(free_space)

    # TODO: add 2 layers
    for _ in range(1):
        elements_list.append(trainable_diffractive_layer)
        elements_list.append(free_space)

    # add Detector in the end of the system!
    elements_list.append(
        Detector(
            simulation_parameters=SIM_PARAMS,
            func='intensity'  # detector that returns intensity
        )
    )

    return LinearOpticalSetup(elements=elements_list)


# TODO: add 507 layers
NUM_OF_DIFF_LAYERS_NO_TRAIN = 2
# TODO: add 253 layers
NUM_OF_DIFF_LAYERS_BEGINNING = 1

optical_setup = set_setup(
    total_number_of_layers=NUM_OF_DIFF_LAYERS_NO_TRAIN,
    number_of_layers_at_the_beginning=NUM_OF_DIFF_LAYERS_BEGINNING
)

CALCULATE_ACCURACIES = True

# create a DetectorProcessorOzcanClf object
if CALCULATE_ACCURACIES:
    detector_processor = DetectorProcessorClf(
        simulation_parameters=SIM_PARAMS,
        num_classes=number_of_classes,
        segmented_detector=DETECTOR_MASK,
    )
else:
    detector_processor = None

# TODO: add 32
train_bs = 256  # a batch size for training set
val_bs = 16  # a batch size for validation set

LR = 1e-3  # learning rate

loss_func_clf = nn.MSELoss()  # by default: reduction='mean'
loss_func_name = 'MSE'


def get_adam_optimizer(net):
    return torch.optim.Adam(
        params=net.parameters(),  # NETWORK PARAMETERS!
        lr=LR
    )


# mnist_wf_train_ds
train_wf_ds, val_wf_ds = torch.utils.data.random_split(
    dataset=mnist_wf_train_ds,
    lengths=[55000, 5000],  # sizes from the article
    generator=torch.Generator().manual_seed(178)  # for reproducibility
)

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

test_wf_loader = torch.utils.data.DataLoader(
    mnist_wf_test_ds,
    batch_size=10,
    shuffle=False,
    # num_workers=2,
    drop_last=False,
)  # data loader for a test MNIST data

import pickle
def onn_train_mse(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR needed for accuracies only!
    loss_func, optimizer,
    device='cuda', show_process=False
):
    """
    Function to train `optical_net` (classification task)
    ...

    Parameters
    ----------
        optical_net : torch.nn.Module
            Neural Network composed of Elements.
        wavefronts_dataloader : torch.utils.data.DataLoader
            A loader (by batches) for the train dataset of wavefronts.
        detector_processor_clf : DetectorProcessorClf
            A processor of a detector image for a classification task, that
            returns `probabilities` of classes.
        loss_func :
            Loss function for a multi-class classification task.
        optimizer: torch.optim
            Optimizer...
        device : str
            Device to computate on...
        show_process : bool
            Flag to show (or not) a progress bar.

    Returns
    -------
        batches_losses : list[float]
            Losses for each batch in an epoch.
        batches_accuracies : list[float]
            Accuracies for each batch in an epoch.
        epoch_accuracy : float
            Accuracy for an epoch.
    """
    optical_net.train()  # activate 'train' mode of a model
    batches_losses = []  # to store loss for each batch
    batches_accuracies = []  # to store accuracy for each batch

    correct_preds = 0
    size = 0

    for batch_wavefronts, batch_targets in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='train', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]

        batch_wavefronts = batch_wavefronts.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        # torch.cuda.memory._record_memory_history(device=device)
        # forward of an optical network
        detector_output = optical_net(batch_wavefronts)

        memory_snapshot = torch.cuda.memory_snapshot()

        with open("memory_snapshot.pickle", "wb") as f:
            pickle.dump(memory_snapshot, f)


        # torch.cuda.memory._snapshot()
        # torch.cuda.memory._dump_snapshot()

        # torch.cuda.memory._record_memory_history(enabled=None)

        # if torch.cuda.is_available():
        #     memory_snapshot = torch.cuda.memory._snapshot()
        #     with open(f"{MEMORY_SNAPSHOTS_FOLDER}/memory_snapshot_forward_pass.pickle", "wb") as f:
        #         pickle.dump(memory_snapshot, f)

        # calculate loss for a batch
        loss = loss_func(detector_output, batch_targets)

        # if torch.cuda.is_available():
        #     memory_snapshot = torch.cuda.memory._snapshot()
        #     with open(f"{MEMORY_SNAPSHOTS_FOLDER}/memory_snapshot_loss_calc.pickle", "wb") as f:
        #         pickle.dump(memory_snapshot, f)

        loss.backward()
        optimizer.step()

        # ACCURACY
        if CALCULATE_ACCURACIES:
            # process a detector image
            batch_labels = detector_processor_clf.batch_forward(batch_targets).argmax(1)    # noqa: E501
            batch_probas = detector_processor_clf.batch_forward(detector_output)    # noqa: E501

            batch_correct_preds = (
                batch_probas.argmax(1) == batch_labels
            ).type(torch.float).sum().item()

            correct_preds += batch_correct_preds
            size += batch_size

        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        if CALCULATE_ACCURACIES:
            batches_accuracies.append(batch_correct_preds / batch_size)
        else:
            batches_accuracies.append(0.)

    if CALCULATE_ACCURACIES:
        epoch_accuracy = correct_preds / size
    else:
        epoch_accuracy = 0.

    return batches_losses, batches_accuracies, epoch_accuracy


def onn_validate_mse(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR NEEDED!
    loss_func,
    device='cuda', show_process=False
):
    """
    Function to validate `optical_net` (classification task)
    ...

    Parameters
    ----------
        optical_net : torch.nn.Module
            Neural Network composed of Elements.
        wavefronts_dataloader : torch.utils.data.DataLoader
            A loader (by batches) for the train dataset of wavefronts.
        detector_processor_clf : DetectorProcessorClf
            A processor of a detector image for a classification task, that
            returns `probabilities` of classes.
        loss_func :
            Loss function for a multi-class classification task.
        device : str
            Device to computate on...
        show_process : bool
            Flag to show (or not) a progress bar.

    Returns
    -------
        batches_losses : list[float]
            Losses for each batch in an epoch.
        batches_accuracies : list[float]
            Accuracies for each batch in an epoch.
        epoch_accuracy : float
            Accuracy for an epoch.
    """
    optical_net.eval()  # activate 'eval' mode of a model
    batches_losses = []  # to store loss for each batch
    batches_accuracies = []  # to store accuracy for each batch

    correct_preds = 0
    size = 0

    for batch_wavefronts, batch_targets in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='validation', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]

        batch_wavefronts = batch_wavefronts.to(device)
        batch_targets = batch_targets.to(device)

        with torch.no_grad():
            detector_outputs = optical_net(batch_wavefronts)
            # calculate loss for a batch
            loss = loss_func(detector_outputs, batch_targets)

        # ACCURACY
        if CALCULATE_ACCURACIES:
            # process a detector image
            batch_labels = detector_processor_clf.batch_forward(batch_targets).argmax(1)    # noqa: E501
            batch_probas = detector_processor_clf.batch_forward(detector_outputs)   # noqa: E501

            batch_correct_preds = (
                batch_probas.argmax(1) == batch_labels
            ).type(torch.float).sum().item()

            correct_preds += batch_correct_preds
            size += batch_size

        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        if CALCULATE_ACCURACIES:
            batches_accuracies.append(batch_correct_preds / batch_size)
        else:
            batches_accuracies.append(0.)

    if CALCULATE_ACCURACIES:
        epoch_accuracy = correct_preds / size
    else:
        epoch_accuracy = 0.

    return batches_losses, batches_accuracies, epoch_accuracy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optical_setup.net = optical_setup.net.to(DEVICE)
SIM_PARAMS = SIM_PARAMS.to(DEVICE)
detector_processor = detector_processor.to(DEVICE)

n_epochs = 1
print_each = 2  # print each n'th epoch info

scheduler = None  # sheduler for a lr tuning during training


# Linc optimizer to a recreated net!
optimizer_clf = get_adam_optimizer(optical_setup.net)

clerk = Clerk('GPU_512_mnist_mse_fix_experiment')

clerk.set_checkpoint_targets({
    'model': optical_setup.net,
    'optimizer': optimizer_clf
})


train_epochs_losses = []
val_epochs_losses = []  # to store losses of each epoch

train_epochs_acc = []
val_epochs_acc = []  # to store accuracies

torch.manual_seed(98)  # for reproducability?
print(DEVICE)
with clerk.begin(
    autosave_checkpoint=True
):
    for epoch in range(n_epochs):
        if (epoch == 0) or ((epoch + 1) % print_each == 0) or (epoch == n_epochs - 1):
            print(f'Epoch #{epoch + 1}: ', end='')
            show_progress = True
        else:
            show_progress = False

        # TRAIN
        start_train_time = time.time()  # start time of the epoch (train)
        train_losses, _, train_accuracy = onn_train_mse(
            optical_setup.net,  # optical network composed
            train_wf_loader,  # dataloader of training set
            detector_processor,  # detector processor
            loss_func_clf,
            optimizer_clf,
            device=DEVICE,
            show_process=show_progress,
        )  # train the model
        mean_train_loss = np.mean(train_losses)

        # train info
        if (epoch == 0) or ((epoch + 1) % print_each == 0) or (epoch == n_epochs - 1):  # noqa: E501
            print('Training results')
            print(f'\t{loss_func_name} : {mean_train_loss:.6f}')
            if CALCULATE_ACCURACIES:
                print(f'\tAccuracy : {(train_accuracy*100):>0.1f} %')
            print(f'\t------------   {time.time() - start_train_time:.2f} s')

        clerk.write_checkpoint(metadata={'epoch': epoch})
        clerk.write_log('loss', {
            'mean_train_loss': mean_train_loss,
            'train_accuracy': train_accuracy*100
        })

        # VALIDATION
        start_val_time = time.time()  # start time of the epoch (validation)
        val_losses, _, val_accuracy = onn_validate_mse(
            optical_setup.net,  # optical network composed in 3.
            val_wf_loader,  # dataloader of validation set
            detector_processor,  # detector processor
            loss_func_clf,
            device=DEVICE,
            show_process=show_progress,
        )  # evaluate the model
        mean_val_loss = np.mean(val_losses)

        # validation info
        if (epoch == 0) or ((epoch + 1) % print_each == 0) or (epoch == n_epochs - 1):  # noqa: E501
            print('Validation results')
            print(f'\t{loss_func_name} : {mean_val_loss:.6f}')
            if CALCULATE_ACCURACIES:
                print(f'\tAccuracy : {(val_accuracy*100):>0.1f} %')
            print(f'\t------------   {time.time() - start_val_time:.2f} s')

        if scheduler:
            scheduler.step(mean_val_loss)

        # save losses
        train_epochs_losses.append(mean_train_loss)
        val_epochs_losses.append(mean_val_loss)
        # save accuracies
        train_epochs_acc.append(train_accuracy)
        val_epochs_acc.append(val_accuracy)

# Загрузка снимка памяти
with open("memory_snapshot.pickle", "rb") as f:
    loaded_snapshot = pickle.load(f)

# Пример анализа
print(f"Количество записей в снимке: {len(loaded_snapshot)}")
print("Пример записи:", loaded_snapshot[0])
# learning curve
fig, axs = plt.subplots(1, 2, figsize=(10, 3))

axs[0].plot(range(1, n_epochs + 1), np.array(train_epochs_losses) * 1e3, label='train')
axs[0].plot(range(1, n_epochs + 1), np.array(val_epochs_losses) * 1e3, linestyle='dashed', label='validation')

axs[1].plot(range(1, n_epochs + 1), train_epochs_acc, label='train')
axs[1].plot(range(1, n_epochs + 1), val_epochs_acc, linestyle='dashed', label='validation')

axs[0].set_ylabel(loss_func_name + r' $\times 10^3$')
axs[0].set_xlabel('Epoch')
axs[0].legend()

axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend()

if not os.path.exists('images'):
    os.makedirs('images')

fig.savefig(f'images/GPU_512_DL_mnist_mse_{DETECTOR_SIZE[0]}x{DETECTOR_SIZE[1]}_grid_losses_and_accuracy.png', dpi=500)  # Сохраняем с высоким разрешением
plt.close(fig)

# array with all losses
# TODO: make with PANDAS?
all_lasses_header = ','.join([
    f'{loss_func_name.split()[0]}_train', f'{loss_func_name.split()[0]}_val',
    'accuracy_train', 'accuracy_val'
])
all_losses_array = np.array(
    [train_epochs_losses, val_epochs_losses, train_epochs_acc, val_epochs_acc]
).T

# Индексы объектов, которые нужно визуализировать
target_indices = {1, 3, 511, 1021, 1023}

# Определяем количество колонок и строк для визуализации
n_cols = len(target_indices)  # Количество колонок равно числу целевых индексов
n_rows = 1

# # Создаем фигуру для визуализации
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.2, n_rows * 4.6))

# cmap = 'rainbow'  # Цветовая карта для визуализации
# count = 1
# # Перебираем слои в optical_setup
# for ind_layer, layer in enumerate(optical_setup.net.to(torch.device("cpu"))):
#     if ind_layer in target_indices and isinstance(layer, elements.DiffractiveLayer):    # noqa: E501

#         # Определяем текущий subplot
#         ax_this = axs[list(target_indices).index(ind_layer)]

#         # Добавляем заголовок с индексом слоя
#         ax_this.set_title(f'DiffractiveLayer {count}')
#         count += 1

#         # Получаем mask для визуализации
#         mask_to_visualize = layer.mask.detach()

#         # Визуализируем mask
#         im = ax_this.imshow(
#             mask_to_visualize, cmap=cmap,
#             vmin=0, vmax=MAX_PHASE
#         )
#         x_frame = (x_layer_nodes - DETECTOR_SIZE[1]) / 2
#         y_frame = (y_layer_nodes - DETECTOR_SIZE[0]) / 2
#         ax_this.set_xlim([x_frame, x_layer_nodes - x_frame])
#         ax_this.set_ylim([y_frame, y_layer_nodes - y_frame])

#         cbar = fig.colorbar(
#             im,
#             ax=ax_this,
#             orientation='vertical',
#             fraction=0.046,
#             pad=0.04
#         )
#         cbar.set_label('Mask Value')
# fig.savefig(f'images/GPU_512_DL_mnist_mse_{DETECTOR_SIZE[0]}x{DETECTOR_SIZE[1]}_masks.png', dpi=500)  # Сохраняем с высоким разрешением
# plt.close(fig)
RESULTS_FOLDER = f'models/reproduced_results/MNIST_MSE_Ozcan_2018-2020_GPU_{512}_DL_{DETECTOR_SIZE[0]}x{DETECTOR_SIZE[1]}_grid'

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# filepath to save the model
model_filepath = f'{RESULTS_FOLDER}/optical_setup_net_gpu.pth'
# filepath to save losses
losses_filepath = f'{RESULTS_FOLDER}/training_curves_gpu.csv'

# saving model
torch.save(optical_setup.net.state_dict(), model_filepath)

# saving losses
np.savetxt(
    losses_filepath, all_losses_array,
    delimiter=',', header=all_lasses_header, comments=""
)

RESULTS_FOLDER = f'models/reproduced_results/MNIST_MSE_Ozcan_2018-2020_GPU_{512}_DL_{DETECTOR_SIZE[0]}x{DETECTOR_SIZE[1]}_grid'

load_model_filepath = f'{RESULTS_FOLDER}/optical_setup_net_gpu.pth'
