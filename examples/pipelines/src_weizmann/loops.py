import torch
from tqdm import tqdm
from torch import nn
from svetlanna import SimulationParameters
from svetlanna import Wavefront


def rnn_train(
    optical_net, wavefronts_dataloader,
    detector_processor_clf,  # DETECTOR PROCESSOR needed for accuracies only!
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
    ind_batch = 1
    
    for batch_wavefronts, batch_targets, batch_labels in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='train', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]
        
        batch_wavefronts = batch_wavefronts.to(device)
        batch_targets = batch_targets.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()

        # forward of an optical network
        detector_outputs = optical_net(batch_wavefronts)

        if LOSS == 'CE':
            # process a detector image
            batch_probas = detector_processor_clf.batch_forward(detector_outputs)
            # calculate loss for a batch
            loss = loss_func(batch_probas, batch_labels)

        if LOSS == 'MSE':
            # calculate loss for a batch
            loss = loss_func(detector_outputs, batch_targets.to(torch.float64))
        
        loss.backward()
        optimizer.step()

        # ACCURACY
        if CALCULATE_ACCURACIES:
            
            if LOSS == 'MSE':
                # process a detector image
                batch_probas = detector_processor_clf.batch_forward(detector_outputs)
            
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


def rnn_validate(
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

    for batch_wavefronts, batch_targets, batch_labels in tqdm(
        wavefronts_dataloader,
        total=len(wavefronts_dataloader),
        desc='validation', position=0,
        leave=True, disable=not show_process
    ):  # go by batches
        # batch_wavefronts - input wavefronts, batch_labels - labels
        batch_size = batch_wavefronts.size()[0]
        
        batch_wavefronts = batch_wavefronts.to(device)
        batch_targets = batch_targets.to(device)
        batch_labels = batch_labels.to(device)
        
        with torch.no_grad():

            detector_outputs = optical_net(batch_wavefronts)

            if LOSS == 'CE':
                # process a detector image
                batch_probas = detector_processor_clf.batch_forward(detector_outputs)
                # calculate loss for a batch
                loss = loss_func(batch_probas, batch_labels)
            
            if LOSS == 'MSE':
                # calculate loss for a batch
                loss = loss_func(detector_outputs, batch_targets)

        # ACCURACY
        if CALCULATE_ACCURACIES:

            if LOSS == 'MSE':
                # process a detector image
                batch_probas = detector_processor_clf.batch_forward(detector_outputs)
                
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