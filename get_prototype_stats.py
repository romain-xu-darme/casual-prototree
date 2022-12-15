import copy
from tqdm import tqdm
from util.args import *
from typing import List
from util.data import get_dataloaders
from prototree.prototree import ProtoTree
from util.save import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from prototree.upsample import find_threshold_to_area, convert_bbox_coordinates
from util.gradients import smoothgrads, prp, cubic_upsampling, normalize_min_max
from features.prp import canonize_tree
import torch
import cv2

# Use only deterministic algorithms
torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Compare different upsampling modes when upsampling prototypes')
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
    parser.add_argument('--proj_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Directory for saving the prototypes visualizations')
    parser.add_argument('--stats_file',
                        type=str,
                        metavar='<name>',
                        required=True,
                        help='Stats file name inside projection directory.')
    parser.add_argument('--target_areas',
                        type=float,
                        nargs='+',
                        metavar='<value>',
                        help='Target bounding box areas')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    add_finalize_args(parser)
    return parser


def compute_prototype_stats(
        tree: ProtoTree,
        img: Image,
        segm: Image,
        img_tensor: torch.Tensor,
        node_id: int,
        img_name: str,
        target_areas: List[float],
        output_dir: str,
        output_filename: str = 'prototype_stats.csv',
        location: Tuple[int, int] = None,
        device: str = 'cuda:0',
        quiet: bool = False,
) -> None:
    """ Compute fidelity and relevance stats for a given prototype

    :param tree: ProtoTree
    :param img: Original image
    :param segm: Image segmentation (if any)
    :param img_tensor: Image tensor
    :param node_id: Node ID = index of the prototype in the similarity map
    :param img_name: Will be used in the statistic file
    :param target_areas: Target bounding box areas
    :param output_dir: Destination folder
    :param output_filename: File name
    :param location: These coordinates are used to determine the upsampling target location
    :param device: Target device
    :param quiet: In quiet mode, does not create images with bounding boxes
    """
    # Mask out original image with segmentation if present
    segm = segm if segm is None else np.asarray(segm)
    img_bgr_uint8 = cv2.cvtColor(np.uint8(np.asarray(img)), cv2.COLOR_RGB2BGR)

    # Use canonized tree for PRP
    canonized_tree = canonize_tree(copy.deepcopy(tree), arch=tree.base_arch, device=device)
    mnames = {smoothgrads: 'smoothgrads', prp: 'prp', cubic_upsampling: 'vanilla'}
    # Vanilla (white), PRP (purple) and Smoothgrads (yellow)
    for method, color in zip([cubic_upsampling, prp, smoothgrads], [(255, 255, 255),(255, 0, 255),(0, 255, 255)]):
        # Compute gradients
        grads = method(
            tree=tree if method != prp else canonized_tree,
            img_tensor=copy.deepcopy(img_tensor),
            node_id=node_id,
            location=location,
        )
        grads = cv2.resize(grads, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
        grads = normalize_min_max(grads)

        img_tensors = [img_tensor.clone().detach()]
        areas = []
        relevances = []
        for target_area in target_areas:
            xmin, xmax, ymin, ymax, effective_area = find_threshold_to_area(grads, target_area)
            areas.append(effective_area)
            if target_area == 0.1:
                # Update result image
                cv2.rectangle(img_bgr_uint8, (xmin, ymin), (xmax-1, ymax-1), color, thickness=2)

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
        h, w = location
        ref_similarity = sim_map[0, h, w]
        fidelities = sim_map[1:, h, w] / ref_similarity

        with open(os.path.join(output_dir, output_filename), 'a') as fout:
            for area, relevance, fidelity in zip(areas, relevances, fidelities):
                fout.write(f'{img_name}, {node_id}, {mnames[method]}, {area}, {relevance}, {fidelity}\n')

    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    if not quiet:
        plt.imsave(os.path.join(output_dir, f'{img_name}.png'), img_rgb_float)


def finalize_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args

    # Obtain the dataset and dataloaders
    _, projectloader, _, _, _ = get_dataloaders(
        dataset=args.dataset,
        projection_mode=args.projection_mode,
        batch_size=args.batch_size,
        device=args.device,
    )
    os.makedirs(args.proj_dir, exist_ok=True)

    # Reset stat file
    open(os.path.join(args.proj_dir, args.stats_file), 'w')

    # Load tree
    tree, _, _, _ = load_checkpoint(args.tree_dir)
    tree.to(args.device)
    # Indicate backbone architecture for PRP canonization
    tree.base_arch = args.base_arch

    # Pruning
    prune(tree, args.pruning_threshold_leaves, None)

    # Projection
    project_info, tree = project_with_class_constraints(tree, projectloader, args.device, None)
    tree.eval()

    # Raw images from projection set
    imgs = projectloader.dataset.imgs
    for node, j in tqdm(tree._out_map.items()):
        if node in tree.branches:  # do not upsample when node is pruned
            prototype_info = project_info[j]
            node_name = prototype_info['node_ix']
            # Open image without preprocessing
            img_path = imgs[prototype_info['input_image_ix']][0]
            x = Image.open(img_path).convert('RGB')
            fname = os.path.splitext(os.path.basename(img_path))[0]
            segm = Image.open(os.path.join(args.segm_dir, img_path.split('/')[-2], fname + '.jpg')).convert('RGB') \
                if args.segm_dir is not None else None
            prototype_location = prototype_info['patch_ix']
            W, H = prototype_info['W'], prototype_info['H']

            compute_prototype_stats(
                tree=tree,
                img=x,
                segm=segm,
                img_tensor=prototype_info['nearest_input'],
                node_id=tree._out_map[node],
                img_name=f'proto_{node_name}',
                output_dir=args.proj_dir,
                output_filename=args.stats_file,
                target_areas=args.target_areas,
                location=(prototype_location // H, prototype_location % H),
                device=args.device,
            )


if __name__ == '__main__':
    finalize_tree()
