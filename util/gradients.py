import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from typing import List, Optional, Tuple
from PIL import Image

from prototree.prototree import ProtoTree


def polarity_and_collapse(
        array: np.array,
        polarity: Optional[str] = None,
        avg_chan: Optional[int] = None,
) -> np.array:
    """ Apply polarity filter (optional) followed by average over channels (optional)

    :param array: Target
    :param polarity: Polarity (positive, negative, absolute)
    :param avg_chan: Dimension across which channels are averaged
    """
    assert polarity in [None, 'positive', 'negative', 'absolute'], f'Invalid polarity {polarity}'

    # Polarity first
    if polarity == 'positive':
        array = np.maximum(0, array)
    elif polarity == 'negative':
        array = np.abs(np.minimum(0, array))
    elif polarity == 'absolute':
        array = np.abs(array)

    # Channel average
    if avg_chan is not None:
        array = np.average(array, axis=avg_chan)
    return array


def normalize_min_max(array: np.array) -> np.array:
    """ Perform min-max normalization of a numpy array

    :param array: Target
    """
    vmin = np.amin(array)
    vmax = np.amax(array)
    # Avoid division by zero
    return (array - vmin) / (vmax - vmin + np.finfo(np.float32).eps)


def smoothgrads(
        tree: ProtoTree,
        sample: torch.Tensor,
        node_id: int,
        device: Optional[str] = 'cpu',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = True,
        nsamples: Optional[int] = 10,
        noise: Optional[float] = 0.2,
) -> Tuple[List[Image.Image], np.array]:
    """ Perform patch visualization using SmoothGrad

    :param tree: Prototree
    :param sample: Input image tensor
    :param node_id: Node index
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param nsamples: Number of samples
    :param noise: Noise level
    :return: gradient map
    """
    # Find location of feature vector closest to target node
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(sample)
    distances_batch = distances_batch[0, node_id].cpu().numpy()  # Shape H x W
    (h, w) = np.where(distances_batch == np.min(distances_batch))
    h, w = h[0], w[0]

    # Compute variance from noise ratio
    sigma = (sample.max() - sample.min()).cpu().numpy() * noise
    # Generate noisy images around original.
    noisy_images = [sample + torch.randn(sample.shape).to(device) * sigma for _ in range(nsamples)]

    # Compute gradients
    grads = []
    for x in noisy_images:
        x.requires_grad_()
        # Forward pass
        _, distances_batch, _ = tree.forward_partial(x)
        # Identify target location before backward pass
        output = distances_batch[0, node_id, h, w]
        output.backward(retain_graph=True)
        grads.append(x.grad.data[0].detach().cpu().numpy())

    # grads has shape (nsamples) x sample.shape => average across all samples
    grads = np.mean(np.array(grads), axis=0)

    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads
