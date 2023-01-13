import torchvision.datasets
import torchvision.transforms as transforms
from prototree.prototree import ProtoTree
from PIL import Image
import argparse
import torch
from tqdm import tqdm
import os
import copy
import cv2
import numpy as np
from util.gradients import smoothgrads, prp, cubic_upsampling, normalize_min_max
from features.prp import canonize_tree

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def compute_inference_heatmaps(
        tree: ProtoTree,
        img: Image,
        img_tensor: torch.Tensor,
        img_path: str,
        output_dir: str,
        device: str = 'cuda:0',
) -> None:
    """ Compute saliency maps for a given image

    :param tree: ProtoTree
    :param img: Original image
    :param img_tensor: Input image tensor
    :param img_path: Path to original image
    :param output_dir: Path to output directory
    :param device: Target device
    """
    # Use canonized tree for PRP
    canonized_tree = canonize_tree(copy.deepcopy(tree), arch=tree.base_arch, device=device)
    mnames = {smoothgrads: 'smoothgrads', prp: 'prp', cubic_upsampling: 'vanilla'}

    # Get the model prediction
    with torch.no_grad():
        pred_kwargs = dict()
        pred, pred_info = tree.forward(img_tensor, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']

    decision_path = tree.path_to(tree.nodes_by_index[pred_info['out_leaf_ix'][0]])
    img_prefix = os.path.splitext(os.path.basename(img_path))[0]

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
                device=device,
            )
            grads = cv2.resize(grads, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
            grads = normalize_min_max(grads)
            grad_path = os.path.join(output_dir, 'db', f'{img_prefix}_{mnames[method]}.npy')
            np.save(grad_path, grads)
            with open(os.path.join(output_dir, "database.csv"), 'a') as fout:
                fout.write(f'{img_path},{node_id},{depth},{mnames[method]},{grad_path}\n')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Given a dataset of test images, '
                                     'compute and store all saliency maps during inference')
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
    parser.add_argument('--device',
                        type=str,
                        metavar='<device>',
                        default='cuda:0',
                        help='Target device')
    parser.add_argument('--output_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Path to output directory.')
    parser.add_argument('--img_size',
                        type=int,
                        metavar='<size>',
                        default=224,
                        help='Image size')
    parser.add_argument('--restart_from',
                        type=str,
                        metavar='<name>',
                        help='Restart from a given image')
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

    os.makedirs(os.path.join(args.output_dir,'db'), exist_ok=True)
    wait = args.restart_from is not None
    if not os.path.exists(os.path.join(args.output_dir,'database.csv')) or not wait:
        with open(os.path.join(args.output_dir,'database.csv'), 'w') as fout:
            fout.write('img_path,node_id,depth,method,grad_path\n')

    stats_iter = tqdm(
        enumerate(img_set),
        total=len(img_set),
        desc='Computing fidelity stats')
    for index, (img, label) in stats_iter:
        img_path = img_set.samples[index][0]
        # Raw file name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if wait and img_name != args.restart_from:
            continue
        wait = False
        compute_inference_heatmaps(
            tree=tree,
            img=img,
            img_tensor=transform(img).unsqueeze(0).to(args.device),
            img_path=img_path,
            output_dir=args.output_dir,
            device=args.device,
        )
