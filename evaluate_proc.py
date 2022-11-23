import torchvision.datasets
import torchvision.transforms as transforms
from prototree.prototree import ProtoTree
from prototree.upsample import upsample_similarity_map
from util.args import *
from PIL import Image
import argparse
import torch
from tqdm import tqdm
import os


def upsample_local(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        img: Image,
        seg: Image,
        decision_path: list,
        threshold: str,
        mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> List[float]:
    """ Given a test sample, compute and store visual representation of parts similar to prototypes

    :param tree: ProtoTree
    :param img_tensor: Input image tensor
    :param img: Original image
    :param seg: Segmentation of the original image
    :param decision_path: List of main nodes leading to the prediction
    :param threshold: Upsampling threshold
    :param mode: Either "vanilla" or "smoothgrads"
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    :returns: List of overlap ratios between the prototypes bounding boxes and the object
    """
    overlaps = []
    for node in decision_path[:-1]:
        overlaps.append(upsample_similarity_map(
            tree=tree,
            img=img,
            seg=seg.convert('RGB'),
            img_tensor=img_tensor,
            node_id=tree._out_map[node],
            node_name=node.index,
            output_dir=None,
            threshold=threshold,
            location=None,  # Upsample location maximizing similarity
            mode=mode,
            grads_x_input=grads_x_input,
        ))
    return overlaps


def get_overlap_stats(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        img: Image,
        seg: Image,
        upsample_threshold: str,
        upsample_mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> Tuple[int, float]:
    """ Generate prediction visualization

    :param tree: ProtoTree
    :param img_tensor: Input image tensor
    :param img: Original image
    :param seg: Segmentation of the original image
    :param upsample_threshold: Upsampling threshold
    :param upsample_mode: Either "vanilla" or "smoothgrads"
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    :returns: Prediction, average percentage of overlap between parts positively compared and object segmentation
    """
    assert upsample_mode in ['vanilla', 'smoothgrads'], f'Unsupported upsample mode {upsample_mode}'

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

    overlaps = upsample_local(
        tree=tree,
        img_tensor=img_tensor,
        img=img,
        seg=seg,
        decision_path=decision_path,
        threshold=upsample_threshold,
        mode=upsample_mode,
        grads_x_input=grads_x_input
    )

    avg_overlap = 1.0
    if overlaps is not None:
        # Compute stats on percentage of overlap between part visualizations and object segmentation
        npos = 0  # Number of positive comparisons
        sum_overlap = 0.0  # Cumulative percentage of overlap
        for i, node in enumerate(decision_path[:-1]):
            node_ix = node.index
            prob = probs[node_ix].item()
            if prob > 0.5:
                sum_overlap += overlaps[i]
                npos += 1
        avg_overlap = sum_overlap/npos if npos else 1.0

    return int(label_ix), avg_overlap


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
        f.write('path;label;pred;overlap\n')
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
                img=img,
                seg=seg,
                upsample_threshold=args.upsample_threshold,
                upsample_mode=args.upsample_mode,
                grads_x_input=args.grads_x_input,
            )
            f.write(f'{img_name};{label};{pred};{stats:.2f}\n')
