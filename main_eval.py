from util.args import *
from util.data import get_dataloaders
from prototree.prototree import ProtoTree
from util.log import Log
from prototree.test import eval_accuracy, eval_fidelity
import torch

# Use only deterministic algorithms
torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Evaluate a ProtoTree')
    add_general_args(parser)
    return parser


def eval_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args
    print('Device used: ', args.device)

    # Load trained ProtoTree
    tree = ProtoTree.load(args.tree_dir, map_location=args.device)

    # Create a logger
    log = Log(args.root_dir, mode='a')
    print("Log dir: ", args.root_dir, flush=True)

    # Obtain the dataloaders
    _, _, testloader, classes, num_channels = get_dataloaders(
        dataset=args.dataset,
        projection_mode=None,
        batch_size=args.batch_size,
        device=args.device,
    )

    eval_accuracy(
        tree=tree,
        test_loader=testloader,
        prefix=args.tree_dir,
        device=args.device,
        log=log,
        sampling_strategy='distributed',
        progress_prefix='Eval '
    )
    eval_accuracy(
        tree=tree,
        test_loader=testloader,
        prefix=args.tree_dir,
        device=args.device,
        log=log,
        sampling_strategy='sample_max',
        progress_prefix='Eval '
    )
    eval_accuracy(
        tree=tree,
        test_loader=testloader,
        prefix=args.tree_dir,
        device=args.device,
        log=log,
        sampling_strategy='greedy',
        progress_prefix='Eval '
    )
    eval_fidelity(tree, testloader, args.device, log)


if __name__ == '__main__':
    eval_tree()
