import torch
from torch.nn import functional
from svetlanna import SimulationParameters


def squares_mnist(
    x_detector_nodes, y_detector_nodes,
    sim_params: SimulationParameters
):
    """
    Returns a mask for a DetectorProcessorClf that can be used for
    a 10-class task (MNIST).
    ...

    Parameters
    ----------
    x_detector_nodes, y_detector_nodes : int
        Size (in nodes) of the zone where all detector zones are located.
        Can be less than (or equal to) a SimulationParameters 'H' and 'W' axes sizes!
    sim_params : SimulationParameters
        Simulations parameters to make a detector mask of the necessary size!

    Returns
    -------
    torch.Tensor
        Mask (`segmented_detector`) for a DetectorProcessorClf.
    """
    # size of each segment
    x_segment_nodes = int(x_detector_nodes / 9)
    y_segment_nodes = int(y_detector_nodes / 9)
    # each segment of size = (y_segment_nodes, x_segment_nodes)

    # coordinates of segments left upper corners (in nodes)
    segments_corners = {}
    
    for ind, label in enumerate(range(3)):
        segments_corners[label] = (
            y_detector_nodes // 4 - y_segment_nodes // 2,
            2 * x_segment_nodes + ind * (2 * x_segment_nodes)
        )
    
    for ind, label in enumerate(range(3, 7)):
        segments_corners[label] = (
            y_detector_nodes // 2 - y_segment_nodes // 2,
            x_segment_nodes + ind * (2 * x_segment_nodes)
        )
    
    for ind, label in enumerate(range(7, 10)):
        segments_corners[label] = (
            3 * y_detector_nodes // 4 - y_segment_nodes // 2,
            2 * x_segment_nodes + ind * (2 * x_segment_nodes)
        )

    # create a mask for detector
    detector_mask = torch.ones(
        size=(y_detector_nodes, x_detector_nodes),
    ) * (-1)
    
    for label in segments_corners.keys():
        y_node, x_node = segments_corners[label]
        detector_mask[y_node:y_node + y_segment_nodes, x_node:x_node + x_segment_nodes] = label
    
    # ADD PADDING IF APERTURES USED!
    y_nodes, x_nodes = sim_params.axes_size(axs=('H', 'W'))
    y_mask, x_mask = detector_mask.size()  # (y_detector_nodes, x_detector_nodes)
    
    if (not y_nodes == y_mask) or (not x_nodes == x_mask):
        # add padding to match simulation parameters size
        # symmetrically! 
        pad_top = int((y_nodes - y_mask) / 2)
        pad_bottom = y_nodes - pad_top - y_mask
        pad_left = int((x_nodes - x_mask) / 2)
        pad_right = x_nodes - pad_left - x_mask  # params for transforms.Pad
        
        # padding transform to match aperture size with simulation parameters     
        detector_mask = functional.pad(
            input=detector_mask,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=-1
        )

    return detector_mask


def circles(
    x_detector_nodes, y_detector_nodes,
    n_classes,
    sim_params: SimulationParameters
):
    """
    Returns a mask for a DetectorProcessorClf of concentric rings of equal thickness.
    ...

    Parameters
    ----------
    x_detector_nodes, y_detector_nodes : int
        Size (in nodes) of the zone where all detector zones are located.
        Can be less than (or equal to) a SimulationParameters 'H' and 'W' axes sizes!
    n_classes : int
        Number of circular zones.
    sim_params : SimulationParameters
        Simulations parameters to make a detector mask of the necessary size!

    Returns
    -------
    torch.Tensor
        Mask (`segmented_detector`) for a DetectorProcessorClf.
    """
    y_nodes, x_nodes = sim_params.axes_size(axs=('H', 'W'))  # y_detector_nodes, x_detector_nodes
    x_layer_size_m = sim_params.axes.W[-1] - sim_params.axes.W[0]
    
    if (not y_nodes == y_detector_nodes) or (not x_nodes == x_detector_nodes):
        max_radii = x_layer_size_m / 2 * x_detector_nodes / x_nodes  # in [m]
    else:
        max_radii = x_layer_size_m / 2  # in [m]
    
    circles_radiuses = torch.linspace(max_radii / n_classes, max_radii, n_classes)

    # empty mask
    detector_circles_mask = torch.ones(
        size=(y_nodes, x_nodes),
    ) * (-1)

    grid_x, grid_y = torch.meshgrid(sim_params.axes['W'], sim_params.axes['H'], indexing='xy')
    distances_mask = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    
    # create a mask for detector
    for ind, radius in enumerate(reversed(circles_radiuses)):
        class_num = n_classes - 1 - ind
        detector_circles_mask[distances_mask <= radius] = class_num
    
    detector_circles_mask[distances_mask > circles_radiuses[-1]] = -1
    
    return detector_circles_mask


def angular_segments(
    x_detector_nodes, y_detector_nodes,
    n_classes,
    sim_params: SimulationParameters
):
    """
    Returns a mask for a DetectorProcessorClf of equal angular segments (dPhi = 2pi / N).
    ...

    Parameters
    ----------
    x_detector_nodes, y_detector_nodes : int
        Size (in nodes) of the zone where all detector zones are located.
        Can be less than (or equal to) a SimulationParameters 'H' and 'W' axes sizes!
    n_classes : int
        Number of circular zones.
    sim_params : SimulationParameters
        Simulations parameters to make a detector mask of the necessary size!

    Returns
    -------
    torch.Tensor
        Mask (`segmented_detector`) for a DetectorProcessorClf.
    """
    class_segment_angle = 2 * torch.pi / n_classes
    
    # empty mask
    detector_mask = torch.ones(
        size=(y_detector_nodes, x_detector_nodes),
    ) * (-1)

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, x_detector_nodes), 
        torch.linspace(-1, 1, y_detector_nodes),  
        indexing='xy'
    )
    angles_mask = torch.arctan2(grid_y, grid_x)
    
    # create a mask for detector
    lower_bound = -torch.pi
    for class_num in range(n_classes):
        upper_bound = lower_bound + class_segment_angle
        detector_mask[(angles_mask >= lower_bound) & (angles_mask < upper_bound)] = class_num

        lower_bound += class_segment_angle  # update left bound for a next class

    # ADD PADDING IF APERTURES USED!
    y_nodes, x_nodes = sim_params.axes_size(axs=('H', 'W'))
    y_mask, x_mask = detector_mask.size()  # (y_detector_nodes, x_detector_nodes)

    if (not y_nodes == y_mask) or (not x_nodes == x_mask):
        # add padding to match simulation parameters size
        # symmetrically! 
        pad_top = int((y_nodes - y_mask) / 2)
        pad_bottom = y_nodes - pad_top - y_mask
        pad_left = int((x_nodes - x_mask) / 2)
        pad_right = x_nodes - pad_left - x_mask  # params for transforms.Pad
        
        # padding transform to match aperture size with simulation parameters     
        detector_mask = functional.pad(
            input=detector_mask,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=-1
        )
    
    return detector_mask

    
