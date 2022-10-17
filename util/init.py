import argparse
import torch
from prototree.prototree import ProtoTree
import pickle


def load_state(directory_path: str, device):
    with open(directory_path + '/tree.pkl', 'rb') as f:
        tree = pickle.load(f)
        state = torch.load(directory_path + '/model_state.pth', map_location=device)
        tree.load_state_dict(state)
    return tree


def init_tree(tree: ProtoTree, optimizer, scheduler, device: str, args: argparse.Namespace):
    epoch = 1
    mean = 0.5
    std = 0.1
    if args.state_dict_dir_net != '':  # load pretrained conv network
        # initialize prototypes
        torch.nn.init.normal_(tree.prototype_layer.prototype_vectors, mean=mean, std=std)
        # strict is False so when loading pretrained model, ignore the linear classification layer
        tree._net.load_state_dict(torch.load(args.state_dict_dir_net + '/model_state.pth'), strict=False)
        tree._add_on.load_state_dict(torch.load(args.state_dict_dir_net + '/model_state.pth'), strict=False)
    else:
        with torch.no_grad():
            # initialize prototypes
            torch.nn.init.normal_(tree.prototype_layer.prototype_vectors, mean=mean, std=std)
            tree._add_on.apply(init_weights_xavier)
    return tree, epoch


def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))


def init_weights_kaiming(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
