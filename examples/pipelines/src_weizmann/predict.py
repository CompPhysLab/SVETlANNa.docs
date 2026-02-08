import torch
from torch import nn
from svetlanna import SimulationParameters
from svetlanna import Wavefront


def predict_video(net, masks_filepath, flip=True, show_process=True):

    action_name = masks_filepath.split('/')[2].split('_')[0]  # depends on a data directory!
    label = ACTION_TO_ID[action_name]  # label for the sequence
    
    file_seqs_ds = WeizmannDsWfSeqs(
        [masks_filepath],
        transforms_for_ds,
        SIM_PARAMS,
        augmentation_horizontally_flip=flip,
        target='detector',
        detector_mask=DETECTOR_MASK,
    )  # item: (image wavefront, target detector tensor, label)

    # sum of all detector images for each sequence of the video
    detector_image = torch.zeros(
        size = SIM_PARAMS.axes_size(
            axs=('H', 'W')
        )
    )
    seqs_pred_labels = []

    if flip:# sum of all detector images for each flipped squence of the video
        flipped_detector_image = torch.zeros(
            size = SIM_PARAMS.axes_size(
                axs=('H', 'W')
            )
        )
        flipped_seqs_pred_labels = []

    ind_this = 0
    for seq_wavefronts, seq_target, seq_label in tqdm(file_seqs_ds, disable=not show_process):
        
        net.eval()  # pedict sequence:
        with torch.no_grad():
            seq_detector = net(seq_wavefronts)
            if flip:
                if ind_this % 2 == 0:  # not flipped sequence
                    detector_image += seq_detector
                else:  # flipped sequence
                    flipped_detector_image += seq_detector
            else:
                detector_image += seq_detector
    
            # process a detector image
            if detector_processor:
                pred_label_this = detector_processor.forward(detector_image).argmax().item()  # predicted label
                
                if flip:
                    if ind_this % 2 == 0:  # not flipped sequence
                        seqs_pred_labels.append(pred_label_this)
                    else:  # flipped sequence
                        flipped_seqs_pred_labels.append(pred_label_this)
                else:
                    seqs_pred_labels.append(pred_label_this)

        ind_this +=1
    if flip:
        return detector_image + flipped_detector_image, seqs_pred_labels, flipped_seqs_pred_labels, label
    else:
        return detector_image, seqs_pred_labels, label
