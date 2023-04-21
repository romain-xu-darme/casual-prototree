import copy
import numpy as np
from tqdm import tqdm
import argparse
from util.args import add_finalize_args, get_args
from typing import Tuple
from util.data import get_dataloaders
from prototree.prototree import ProtoTree
from PIL import Image
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from util.gradients import smoothgrads, prp, cubic_upsampling, normalize_min_max
from features.prp import canonize_tree
import torch
import cv2
import os

# Use only deterministic algorithms
torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Generate saliency maps for prototypes')
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
                        help='Directory for saving the prototypes saliency maps')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    add_finalize_args(parser)
    return parser


def compute_prototype_heatmaps(
        tree: ProtoTree,
        img: Image,
        img_tensor: torch.Tensor,
        img_path: str,
        node_id: int,
        depth: int,
        output_dir: str,
        location: Tuple[int, int] = None,
        device: str = 'cuda:0',
) -> None:
    """ Compute saliency maps of a given prototype

    :param tree: ProtoTree
    :param img: Original image
    :param img_tensor: Image tensor
    :param img_path: Path to original image
    :param node_id: Node ID = index of the prototype in the similarity map
    :param depth: Node depth inside the tree
    :param output_dir: Destination folder
    :param location: These coordinates are used to determine the upsampling target location
    :param device: Target device
    """
    # Use canonized tree for PRP
    canonized_tree = canonize_tree(copy.deepcopy(tree), arch=tree.base_arch, device=device)
    mnames = {smoothgrads: 'smoothgrads', prp: 'prp', cubic_upsampling: 'vanilla'}

    img_prefix = os.path.splitext(os.path.basename(img_path))[0]

    # Vanilla (white), PRP (purple) and Smoothgrads (yellow)
    for method in [cubic_upsampling, prp, smoothgrads]:
        # Compute gradients
        grads = method(
            tree=tree if method != prp else canonized_tree,
            img_tensor=copy.deepcopy(img_tensor),
            node_id=node_id,
            location=location,
            device=device,
        )
        grads = cv2.resize(grads, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
        grads = normalize_min_max(grads)
        grad_path = os.path.join(output_dir,'db',f'{img_prefix}_{mnames[method]}.npy')
        np.save(grad_path, grads)
        with open(os.path.join(output_dir, "database.csv"), 'a') as fout:
            fout.write(f'{node_id},{depth},{img_path},{mnames[method]},{grad_path},{location[0]},{location[1]}\n')


def finalize_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args

    # Obtain the dataset and dataloaders
    _, projectloader, _, _, _ = get_dataloaders(
        dataset=args.dataset,
        projection_mode=args.projection_mode,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Reset database file
    os.makedirs(os.path.join(args.proj_dir, 'db'), exist_ok=True)
    with open(os.path.join(args.proj_dir, "database.csv"), 'w') as fout:
        fout.write('node_id,depth,img_path,method,grad_path,h,w\n')

    # Load tree
    tree = ProtoTree.load(args.tree_dir, map_location=args.device)
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
            # Open image without preprocessing
            img_path = imgs[prototype_info['input_image_ix']][0]
            x = Image.open(img_path).convert('RGB')
            prototype_location = prototype_info['patch_ix']
            W, H = prototype_info['W'], prototype_info['H']

            compute_prototype_heatmaps(
                tree=tree,
                img=x,
                img_tensor=prototype_info['nearest_input'],
                img_path=img_path,
                node_id=tree._out_map[node],
                depth=len(tree.path_to(node)),
                output_dir=args.proj_dir,
                location=(prototype_location // H, prototype_location % H),
                device=args.device,
            )

if __name__ == '__main__':
    finalize_tree()
