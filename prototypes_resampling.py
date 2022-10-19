from util.data import get_dataloaders
from util.visualize import gen_vis
from util.save import *
from util.analyse import average_distance_nearest_image
from util.args import *
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from prototree.upsample import upsample, upsample_with_smoothgrads
import torch
import argparse

torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Resample prototypes from a trained ProtoTree')
    add_general_args(parser)
    add_training_args(parser)
    return parser


def prune_and_project():
    args = get_args(create_parser())
    if not args.checkpoint:
        raise ValueError('Missing path to training checkpoint.')

    # Create/open existing logger
    log = Log(args.log_dir, mode='a')
    print("Log dir: ", args.log_dir, flush=True)

    # GPU management
    if not args.disable_cuda and torch.cuda.is_available():
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    log.log_message('Device used: ' + device)

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)

    print('Opening checkpoint ' + args.checkpoint)
    tree, _, _, _ = load_checkpoint(args.checkpoint)

    # Prune
    prune(tree, args.pruning_threshold_leaves, log)
    # Project
    project_info, tree = project_with_class_constraints(tree, projectloader, device, args, log)
    average_distance_nearest_image(project_info, tree, log)

    # Upsample prototype for visualization
    if args.smoothgrads:
        upsample_with_smoothgrads(tree, project_info, projectloader, "pruned_and_projected_sm", args, log)
        gen_vis(tree, "pruned_and_projected_sm", args, classes)
    else:
        upsample(tree, project_info, projectloader, "pruned_and_projected", args, log)
        gen_vis(tree, "pruned_and_projected", args, classes)


if __name__ == '__main__':
    prune_and_project()
