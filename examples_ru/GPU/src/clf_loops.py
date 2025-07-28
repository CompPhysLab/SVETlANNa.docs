import torch
from torch import nn
from svetlanna.detector import DetectorProcessorClf
from tqdm import tqdm


def onn_train_clf(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR NEEDED!
    loss_func, optimizer,
    device='cpu', show_process=False
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
            A processor of a detector image for a classification task, that returns `probabilities` of classes.
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
    
    for batch_wavefronts, batch_labels in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='train', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]
        
        batch_wavefronts = batch_wavefronts.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()

        # forward of an optical network
        detector_output = optical_net(batch_wavefronts)
        
        # process a detector image
        batch_probas = detector_processor_clf.batch_forward(detector_output)
        
        # calculate loss for a batch
        loss = loss_func(batch_probas, batch_labels)
        
        loss.backward()
        optimizer.step()

        # accuracy
        batch_correct_preds = (
            batch_probas.argmax(1) == batch_labels
        ).type(torch.float).sum().item()
        
        correct_preds += batch_correct_preds    
        size += batch_size
        
        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        batches_accuracies.append(batch_correct_preds / batch_size)

    epoch_accuracy = correct_preds / size
    
    return batches_losses, batches_accuracies, epoch_accuracy


def onn_validate_clf(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR NEEDED!
    loss_func,
    device='cpu', show_process=False
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
            A processor of a detector image for a classification task, that returns `probabilities` of classes.
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

    for batch_wavefronts, batch_labels in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='validation', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]
        
        batch_wavefronts = batch_wavefronts.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            detector_output  = optical_net(batch_wavefronts)
            # process a detector image
            batch_probas = detector_processor_clf.batch_forward(detector_output)
            # calculate loss for a batch
            loss = loss_func(batch_probas, batch_labels)

        # accuracy
        batch_correct_preds = (
            batch_probas.argmax(1) == batch_labels
        ).type(torch.float).sum().item()
        
        correct_preds += batch_correct_preds    
        size += batch_size
        
        # accumulate losses and accuracies for batches
        batches_losses.append(loss.item())
        batches_accuracies.append(batch_correct_preds / batch_size)

    epoch_accuracy = correct_preds / size
    
    return batches_losses, batches_accuracies, epoch_accuracy


def onn_validate_clf_with_energy(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR NEEDED!
    loss_func,
    device='cpu', show_process=False
    ):
    """
    Function to validate `optical_net` (classification task)
    Addition:
        (1) calculates energy efficiency for each sample
                efficiency target = int(intensity, target zone) / detector_energy
                efficiency zones  = int(intensity, all zones  ) / detector_energy,
            where detector_energy = int(intensity, all detector).
        (2) calculates averaged detector pictures for each class
                sum(detector pictures, target = i)
    ...
    
    Parameters
    ----------
        optical_net : torch.nn.Module
            Neural Network composed of Elements.
        wavefronts_dataloader : torch.utils.data.DataLoader
            A loader (by batches) for the train dataset of wavefronts.
        detector_processor_clf : DetectorProcessorClf
            A processor of a detector image for a classification task, that returns `probabilities` of classes.
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
    pass