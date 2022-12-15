import torchvision.datasets
import torchvision.transforms as transforms
from prototree.prototree import ProtoTree
from typing import List, Union
from PIL import Image
import argparse
import torch
from tqdm import tqdm
import os
import copy
import cv2
import numpy as np
from prototree.upsample import find_threshold_to_area, convert_bbox_coordinates
from util.gradients import smoothgrads, prp, cubic_upsampling, normalize_min_max
from features.prp import canonize_tree


def compute_inference_stats(
        tree: ProtoTree,
        img: Image.Image,
        segm: Union[Image.Image, None],
        img_tensor: torch.Tensor,
        label: int,
        img_name: str,
        target_areas: List[float],
        output: str,
        device: str,
) -> None:
    """ Compute fidelity and relevance statistics for a given image

    :param tree: ProtoTree
    :param img: Original image
    :param segm: Image segmentation (if any)
    :param img_tensor: Input image tensor
    :param label: Ground truth label
    :param img_name: Image name
    :param target_areas: Find threshold such that bounding boxes cover a given area
    :param output: Path to output file
    :param device: Target device
    :returns: Prediction, fidelity statistics for each node used in positive reasoning
    """
    segm = segm if segm is None else np.asarray(segm)

    # Use canonized tree for PRP
    canonized_tree = canonize_tree(copy.deepcopy(tree), arch=tree.base_arch, device=device)
    mnames = {smoothgrads: 'smoothgrads', prp: 'prp', cubic_upsampling: 'vanilla'}

    # Get the model prediction
    with torch.no_grad():
        pred_kwargs = dict()
        pred, pred_info = tree.forward(img_tensor, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()

    decision_path = tree.path_to(tree.nodes_by_index[pred_info['out_leaf_ix'][0]])

    for depth, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()
        if prob <= 0.5:
            continue  # Ignore negative comparisons
        # Compute gradient map for this node
        node_id = tree._out_map[node]

        # Vanilla (white), PRP (purple) and Smoothgrads (yellow)
        for method in [cubic_upsampling, prp, smoothgrads]:
            # Compute gradients
            grads = method(
                tree=tree if method != prp else canonized_tree,
                img_tensor=copy.deepcopy(img_tensor),
                node_id=node_id,
                location=None,
            )
            grads = cv2.resize(grads, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
            grads = normalize_min_max(grads)

            img_tensors = [img_tensor.clone().detach()]
            areas = []
            relevances = []
            for target_area in target_areas:
                xmin, xmax, ymin, ymax, effective_area = find_threshold_to_area(grads, target_area)
                areas.append(effective_area)

                # Measure intersection with segmentation (if provided)
                relevance = np.sum(np.sum(segm[ymin:ymax, xmin:xmax], axis=2) > 0) if segm is not None else 0
                relevance /= ((ymax - ymin) * (xmax - xmin))
                relevances.append(relevance)

                # Accumulate perturbed images (will be processed in batch)
                # WARNING: bounding box coordinates have been computed on original image dimension, we need to convert them
                xmin_r, xmax_r, ymin_r, ymax_r = convert_bbox_coordinates(
                    xmin, xmax, ymin, ymax,
                    img.width, img.height,
                    img_tensor.size(2), img_tensor.size(3),
                )
                deleted_img = img_tensor.clone().detach()
                deleted_img[0, :, ymin_r:ymax_r, xmin_r:xmax_r] = 0
                img_tensors.append(deleted_img)

            # Compute fidelities
            img_tensors = torch.cat(img_tensors, dim=0)
            with torch.no_grad():
                _, distances_batch, _ = tree.forward_partial(img_tensors)
                sim_map = torch.exp(-distances_batch[:, node_id]).cpu().numpy()
            # Find reference location of most similar patch
            h, w = np.where(sim_map[0] == np.max(sim_map[0]))
            h, w = h[0], w[0]
            ref_similarity = sim_map[0, h, w]
            fidelities = sim_map[1:, h, w] / ref_similarity

            with open(output, 'a') as fout:
                for area, relevance, fidelity in zip(areas, relevances, fidelities):
                    fout.write(f'{img_name}, {label}, {int(label_ix)}, {node_id}, {depth}, '
                               f'{mnames[method]}, {area}, {relevance}, {fidelity}\n')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Given a dataset of test images, '
                                     'evaluate the average fidelity and relevance score')
    parser.add_argument('--tree_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='The directory containing a state dict (checkpoint) with a pretrained prototree. ')
    parser.add_argument('--base_arch',
                        type=str,
                        metavar='<arch>',
                        required=True,
                        help='Architecture of feature extractor (for PRP).')
    parser.add_argument('--dataset',
                        type=str,
                        metavar='<name>',
                        required=True,
                        help='Data set on which the ProtoTree has been trained')
    parser.add_argument('--segm_dir',
                        type=str,
                        metavar='<path>',
                        help='Directory to segmentation of train images (if available)')
    parser.add_argument('--batch_size',
                        type=int,
                        metavar='<num>',
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--device',
                        type=str,
                        metavar='<device>',
                        default='cuda:0',
                        help='Target device')
    parser.add_argument('--output',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Path to stats file.')
    parser.add_argument('--target_areas',
                        type=float,
                        nargs='+',
                        metavar='<value>',
                        help='Target bounding box areas')
    parser.add_argument('--img_size',
                        type=int,
                        metavar='<size>',
                        default=224,
                        help='Image size')
    parser.add_argument('--restart_from',
                        type=int,
                        metavar='<index>',
                        help='Restart from a given image index')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = get_args()

    # Prepare preprocessing
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    img_set = torchvision.datasets.ImageFolder(args.dataset, transform=None)

    # Load tree
    tree = ProtoTree.load(args.tree_dir, map_location=args.device)
    # Indicate backbone architecture for PRP canonization
    tree.base_arch = args.base_arch
    tree.eval()

    if not os.path.exists(args.output):
        with open(args.output, 'w') as f:
            f.write('path, label, pred, node id, depth, method, area, relevance, fidelity\n')

    wait = args.restart_from is not None

    stats_iter = tqdm(
        enumerate(img_set),
        total=len(img_set),
        desc='Computing fidelity stats')
    for index, (img, label) in stats_iter:
        if wait and index != args.restart_from:
            continue
        wait = False
        img_path = img_set.samples[index][0]
        # Raw file name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        segm = Image.open(os.path.join(args.segm_dir, img_path.split('/')[-2], img_name + '.png')).convert('RGB') \
            if args.segm_dir is not None else None
        compute_inference_stats(
            tree=tree,
            img=img,
            segm=segm,
            img_tensor=transform(img).unsqueeze(0).to(args.device),
            label=label,
            img_name=img_name,
            target_areas=args.target_areas,
            output=args.output,
            device=args.device,
        )
