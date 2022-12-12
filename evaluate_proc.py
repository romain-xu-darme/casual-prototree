import torchvision.datasets
import torchvision.transforms as transforms
from prototree.prototree import ProtoTree
from util.args import *
from typing import List, Tuple
from PIL import Image
import argparse
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
from prototree.upsample import find_high_activation_crop
from util.gradients import cubic_upsampling, smoothgrads, prp, normalize_min_max


def get_overlap_stats(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        seg: Image,
        thresholds: List[float],
        upsample_mode: str = 'vanilla',
) -> Tuple[int, List[Tuple[int, float]]]:
    """ Generate prediction visualization

    :param tree: ProtoTree
    :param img_tensor: Input image tensor
    :param seg: Segmentation of the original image
    :param thresholds: Upsampling threshold
    :param upsample_mode: Either "vanilla" or "smoothgrads"
    :returns: Prediction, overlap statistics for each node used in positive reasoning
    """
    assert upsample_mode in ['vanilla', 'smoothgrads', 'prp'], f'Unsupported upsample mode {upsample_mode}'

    # Preprocess segmentation
    seg = np.asarray(seg.convert('RGB'))
    img_size = seg.shape[:2]

    # Get the model prediction
    with torch.no_grad():
        pred_kwargs = dict()
        pred, pred_info = tree.forward(img_tensor, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()
        assert 'out_leaf_ix' in pred_info.keys()

    leaf_ix = pred_info['out_leaf_ix'][0]
    leaf = tree.nodes_by_index[leaf_ix]
    decision_path = tree.path_to(leaf)
    stats = []
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()
        if prob <= 0.5:
            continue  # Ignore negative comparisons

        # Compute stats on percentage of overlap between part visualizations and object segmentation
        node_id = tree._out_map[node]
        if upsample_mode == 'vanilla':
            grads = cubic_upsampling(
                tree=tree,
                img_tensor=img_tensor,
                node_id=node_id,
                location=None,
            )
        elif upsample_mode == 'prp':
            grads = prp(
                tree=tree,
                img_tensor=img_tensor,
                node_id=node_id,
                location=None,
                device=img_tensor.device,
                normalize=False,
                gaussian_ksize=5,
            )
        else:  # Smoothgrads
            grads = smoothgrads(
                tree=tree,
                img_tensor=img_tensor,
                node_id=node_id,
                location=None,
                device=img_tensor.device,
                normalize=False
            )
        grads = cv2.resize(grads, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
        grads = normalize_min_max(grads)

        for threshold in thresholds:
            # For each threshold value, recompute the bounding box and the area of overlap
            high_act_patch_indices = find_high_activation_crop(grads, threshold)
            ymin, ymax = high_act_patch_indices[0], high_act_patch_indices[1]
            xmin, xmax = high_act_patch_indices[2], high_act_patch_indices[3]
            # Measure how much this bounding box intersects with the object
            overlap = np.sum(np.sum(seg[ymin:ymax, xmin:xmax], axis=2) > 0)
            bbox_area = ((ymax - ymin) * (xmax - xmin))
            img_area = grads.shape[0]*grads.shape[1]
            overlap /= bbox_area
            # [ depth in the tree, threshold value, bbox area, percentage of overlap ]
            stats.append([i, threshold, bbox_area/img_area, overlap])

    return int(label_ix), stats


def get_local_expl_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Given a dataset of test images and their segmentation, '
                                     'evaluate the average Positive Reasoning Overlap Coefficient (PROC)')
    add_general_args(parser)
    parser.add_argument('--img_dir',
                        required=True,
                        type=str,
                        metavar='<path>',
                        help='Directory of test images')
    parser.add_argument('--seg_dir',
                        required=True,
                        type=str,
                        metavar='<path>',
                        help='Directory to segmentation of test images')
    parser.add_argument('--thresholds',
                        required=True,
                        type=float, nargs='+',
                        metavar='<value>',
                        help='List of thresholding values')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        metavar='<path>',
                        help='Path to output CSV file')
    parser.add_argument('--img_size',
                        type=int,
                        metavar='<size>',
                        default=224,
                        help='Image size (default: 224)')
    parsed_args = parser.parse_args()
    if not parsed_args.tree_dir:
        parser.error('Missing path to Prototree (--tree_dir')
    return parsed_args


if __name__ == '__main__':
    args = get_local_expl_args()

    # Log which device was actually used
    print('Device used: ', args.device)

    # Load trained ProtoTree
    tree = ProtoTree.load(args.tree_dir, map_location=args.device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(size=(args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    img_set = torchvision.datasets.ImageFolder(args.img_dir)
    seg_set = torchvision.datasets.ImageFolder(args.seg_dir)

    with open(args.output, 'a') as f:
        f.write('path;label;pred;depth;thres;area;overlap\n')
        stats_iter = tqdm(
            enumerate(zip(img_set, seg_set)),
            total=len(img_set),
            desc='Computing overlap stats')
        for index, ((img, label), (seg, _)) in stats_iter:
            # Sanity check to make sure that image and segmentation are matching
            img_name = os.path.splitext(os.path.basename(img_set.samples[index][0]))[0]
            seg_name = os.path.splitext(os.path.basename(seg_set.samples[index][0]))[0]
            assert img_name == seg_name

            pred, stats = get_overlap_stats(
                tree=tree,
                img_tensor=transform(img).unsqueeze(0).to(args.device),
                seg=seg,
                thresholds=args.thresholds,
                upsample_mode=args.upsample_mode,
            )
            for stat in stats:
                f.write(f'{img_name};{label};{pred};{stat[0]};{stat[1]:.1f};{stat[2]:.2f};{stat[3]:.2f}\n')
            f.flush()
