
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
from subprocess import check_call
import math
from PIL import Image
from prototree.upsample import find_high_activation_crop, imsave_with_bbox, upsample_similarity_map
from prototree.upsample import smoothgrads_upsample
import torch

import torchvision
from torchvision.utils import save_image

from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node

from util.gradients import smoothgrads, normalize_min_max
from skimage.filters import threshold_otsu


def smoothgrads_local(
        tree: ProtoTree,
        sample: torch.Tensor,
        sample_dir: str,
        folder_name: str,
        img_name: str,
        decision_path: list,
        args: argparse.Namespace):

    dir = os.path.join(os.path.join(os.path.join(args.log_dir, folder_name), img_name),
                       args.dir_for_saving_images+'_smoothgrads')
    if not os.path.exists(dir):
        os.makedirs(dir)

    img = Image.open(sample_dir)

    for i, node in enumerate(decision_path[:-1]):
        smoothgrads_upsample(
            tree=tree, img=img, img_tensor=sample, node=node, location=None, img_dir=dir, args=args)


def upsample_local(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 img_name: str,
                 decision_path: list,
                 args: argparse.Namespace):

    dir = os.path.join(os.path.join(os.path.join(args.log_dir, folder_name), img_name), args.dir_for_saving_images)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(sample)
        sim_map = torch.exp(-distances_batch[0, :, :, :]).cpu().numpy()
    img = Image.open(sample_dir)
    for node in decision_path[:-1]:
        upsample_similarity_map(
            img=img,
            similarity_map=sim_map[tree._out_map[node]],
            decision_node_idx=node.index,
            img_dir=dir,
            args=args,
        )

def gen_pred_vis(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 args: argparse.Namespace,
                 classes: tuple,
                 pred_kwargs: dict = None,
                 ):
    pred_kwargs = pred_kwargs or dict()  # TODO -- assert deterministic routing

    # Create dir to store visualization
    img_name = sample_dir.split('/')[-1].split(".")[-2]

    if not os.path.exists(os.path.join(args.log_dir, folder_name)):
        os.makedirs(os.path.join(args.log_dir, folder_name))
    destination_folder=os.path.join(os.path.join(args.log_dir, folder_name),img_name)

    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    # Get references to where source files are stored
    upsample_path = os.path.join(os.path.join(args.log_dir,args.dir_for_saving_images),'pruned_and_projected')
    nodevis_path = os.path.join(args.log_dir,'pruned_and_projected/node_vis')
    local_upsample_path = os.path.join(destination_folder, args.dir_for_saving_images)
    if args.use_smoothgrads:
        local_upsample_path += "_smoothgrads"

    # Get the model prediction
    with torch.no_grad():
        pred, pred_info = tree.forward(sample, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()
        assert 'out_leaf_ix' in pred_info.keys()

    # Save input image
    sample_path = destination_folder + '/node_vis/sample.jpg'
    # save_image(sample, sample_path)
    Image.open(sample_dir).save(sample_path)

    # Save an image containing the model output
    output_path = destination_folder + '/node_vis/output.jpg'
    leaf_ix = pred_info['out_leaf_ix'][0]
    leaf = tree.nodes_by_index[leaf_ix]
    decision_path = tree.path_to(leaf)

    if args.use_smoothgrads:
        smoothgrads_local(tree, sample, sample_dir, folder_name, img_name, decision_path, args)
    else:
        upsample_local(tree, sample, sample_dir, folder_name, img_name, decision_path, args)

    # Prediction graph is visualized using Graphviz
    # Build dot string
    s = 'digraph T {margin=0;rankdir=LR\n'
    # s += "subgraph {"
    s += 'node [shape=plaintext, label=""];\n'
    s += 'edge [penwidth="0.5"];\n'

    # Create a node for the sample image
    s += f'sample[image="{sample_path}"];\n'

    # Create nodes for all decisions/branches
    # Starting from the leaf
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()

        s += f'node_{i+1}[image="{upsample_path}/{node_ix}_nearest_patch_of_image.png" group="{"g"+str(i)}"];\n'
        if prob > 0.5:
            s += f'node_{i+1}_original[image="{local_upsample_path}/{node_ix}_bounding_box_nearest_patch_of_image.png" imagescale=width group="{"g"+str(i)}"];\n'
            label = "Present      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        else:
            s += f'node_{i+1}_original[image="{sample_path}" group="{"g"+str(i)}"];\n'
            label = "Absent      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        # s += f'node_{i+1}_original->node_{i+1} [label="{label}" fontsize=10 fontname=Helvetica];\n'

        s += f'node_{i+1}->node_{i+2};\n'
        s += "{rank = same; "f'node_{i+1}_original'+"; "+f'node_{i+1}'+"};"

    # Create a node for the model output
    s += f'node_{len(decision_path)}[imagepos="tc" imagescale=height image="{nodevis_path}/node_{leaf_ix}_vis.jpg" label="{classes[label_ix]}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'

    # Connect the input image to the first decision node
    s += 'sample->node_1;\n'


    s += '}\n'

    pname = "predvis" if not args.use_smoothgrads else "predvis_sm"
    with open(os.path.join(destination_folder, f'{pname}.dot'), 'w') as f:
        f.write(s)

    from_p = os.path.join(destination_folder, f'{pname}.dot')
    to_pdf = os.path.join(destination_folder, f'{pname}.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s' % (from_p, to_pdf), shell=True)


