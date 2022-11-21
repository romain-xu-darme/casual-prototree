import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from prototree.prototree import ProtoTree
from util.log import Log
from util.gradients import cubic_upsampling, smoothgrads, normalize_min_max
from skimage.filters import threshold_otsu
from typing import Tuple


# adapted from protopnet
def upsample_prototypes(
        tree: ProtoTree,
        project_info: dict,
        project_loader: DataLoader,
        output_dir: str,
        threshold: str,
        log: Log,
        mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> None:
    """ Given the projection information, create and store visual representations of prototypes

    :param tree: ProtoTree
    :param project_info: Projection information
    :param project_loader: Dataloader containing all original projection images
    :param output_dir: Output directory
    :param threshold: Threshold for bounding box extraction
    :param log: Logger
    :param mode: Upsampling mode. Either 'vanilla' (cubic interpolation) or 'smoothgrads'
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    """
    assert mode in ['vanilla', 'smoothgrads'], f'Unsupported upsampling mode {mode}'

    os.makedirs(output_dir, exist_ok=True)
    log.log_message("\nUpsampling prototypes for visualization...")
    imgs = project_loader.dataset.imgs
    for node, j in tree._out_map.items():
        if node in tree.branches:  # do not upsample when node is pruned
            prototype_info = project_info[j]
            node_name = prototype_info['node_ix']
            x = Image.open(imgs[prototype_info['input_image_ix']][0])
            x.save(os.path.join(output_dir, '%s_original_image.png' % str(node_name)))

            prototype_location = prototype_info['patch_ix']
            W, H = prototype_info['W'], prototype_info['H']

            upsample_similarity_map(
                tree=tree,
                img=x,
                img_tensor=prototype_info['nearest_input'],
                node_id=tree._out_map[node],
                node_name=node_name,
                output_dir=output_dir,
                threshold=threshold,
                location=(prototype_location // H, prototype_location % H),
                mode=mode,
                grads_x_input=grads_x_input,
            )


def upsample_similarity_map(
        tree: ProtoTree,
        img: Image,
        seg: Image,
        img_tensor: torch.Tensor,
        node_id: int,
        node_name: int,
        output_dir: str,
        threshold: str,
        location: Tuple[int, int] = None,
        mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> float:
    """ Create a visualization of the most similar patch of an image w.r.t. a given prototype

    :param tree: ProtoTree
    :param img: Original image
    :param seg: Image segmentation (if any)
    :param img_tensor: Image tensor
    :param node_id: Node ID = index of the prototype in the similarity map
    :param node_name: Node name (/= node ID)
    :param output_dir: Destination folder
    :param threshold: Threshold for bounding box extraction
    :param location: If given, these coordinates are used to determine the upsampling target location
    :param mode: Upsampling mode. Either 'vanilla' (cubic interpolation) or 'smoothgrads'
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    :returns: ratio of overlap between prototype bounding box and object when segmentation is given, 0 otherwise
    """
    assert mode in ['vanilla', 'smoothgrads'], f'Unsupported upsampling mode {mode}'
    os.makedirs(output_dir, exist_ok=True)

#    img = img if seg is None else ImageChops.multiply(img, seg)
    seg = seg if seg is None else np.asarray(seg)
    x_np = np.asarray(img)
    x_np = np.float32(x_np) / 255
    if x_np.ndim == 2:  # convert grayscale to RGB
        x_np = np.stack((x_np,) * 3, axis=-1)
    img_size = x_np.shape[:2]

    if mode == 'vanilla':
        grads = cubic_upsampling(
            tree=tree,
            img_tensor=img_tensor,
            node_id=node_id,
            location=location,
        )
    else:  # Smoothgrads
        grads = smoothgrads(
            tree=tree,
            img_tensor=img_tensor,
            node_id=node_id,
            location=location,
            device=img_tensor.device,
            normalize=True
        )
    grads = cv2.resize(grads, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = normalize_min_max(grads)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    plt.imsave(
        fname=os.path.join(output_dir, '%s_heatmap.png' % str(node_name)),
        arr=heatmap,
        vmin=0.0, vmax=1.0)
    overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
    plt.imsave(
        fname=os.path.join(output_dir, '%s_heatmap_original_image.png' % str(node_name)),
        arr=overlayed_original_img,
        vmin=0.0, vmax=1.0)

    threshold = 1-threshold_otsu(grads) if threshold == "auto" else float(threshold)
    high_act_patch_indices = find_high_activation_crop(grads, threshold)
    ymin, ymax = high_act_patch_indices[0], high_act_patch_indices[1]
    xmin, xmax = high_act_patch_indices[2], high_act_patch_indices[3]

    # Measure how much this bounding box intersects with the object
    overlap = sum([seg[y, x].any() != 0 for y in range(ymin, ymax) for x in range(xmin, xmax)]) if seg is not None \
        else 0
    overlap /= ((ymax-ymin)*(xmax-xmin))

    high_act_patch = x_np[ymin:ymax, xmin:xmax, :]
    if grads_x_input:
        # Expand dimension and filter out low activations
        grads = np.expand_dims(grads, axis=-1)
        grads *= (grads > threshold_otsu(grads))
        high_act_patch = high_act_patch * grads[ymin:ymax, xmin:xmax]
    plt.imsave(
        fname=os.path.join(output_dir, '%s_nearest_patch_of_image.png' % str(node_name)),
        arr=high_act_patch,
        vmin=0.0, vmax=1.0)

    # In refined mode, mask out low gradients using the heatmap
    img_rgb = x_np * grads if grads_x_input else x_np
    # save the original image with bounding box showing high activation patch
    imsave_with_bbox(
        fname=os.path.join(output_dir, '%s_bounding_box_nearest_patch_of_image.png' % str(node_name)),
        img_rgb=img_rgb,
        bbox_height_start=high_act_patch_indices[0],
        bbox_height_end=high_act_patch_indices[1],
        bbox_width_start=high_act_patch_indices[2],
        bbox_width_end=high_act_patch_indices[3],
        color=(0, 255, 255))
    imsave_with_bbox(
        fname=os.path.join(output_dir, '%s_bounding_box_and_heatmap.png' % str(node_name)),
        img_rgb=overlayed_original_img,
        bbox_height_start=high_act_patch_indices[0],
        bbox_height_end=high_act_patch_indices[1],
        bbox_width_start=high_act_patch_indices[2],
        bbox_width_end=high_act_patch_indices[3],
        color=(0, 255, 255))
    return overlap

# copied from protopnet
def find_high_activation_crop(mask, threshold):
    threshold = 1.-threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


# copied from protopnet
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    plt.imsave(fname, img_rgb_float)
