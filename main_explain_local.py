from prototree.prototree import ProtoTree
from util.data import get_dataloaders
from util.visualize_prediction import gen_pred_vis
from util.args import *
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy
import os


def get_local_expl_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Explain a prediction')
    add_general_args(parser)
    parser.add_argument('--prototree',
                        type=str,
                        help='Directory to trained ProtoTree')
    parser.add_argument('--sample_dir',
                        type=str,
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--results_dir',
                        type=str,
                        default='local_explanations',
                        help='Directory where local explanations will be saved')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Resize images to this size')
    return parser.parse_args()


def explain_local(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print('Device used: ', str(device))

    # Load trained ProtoTree
    tree = ProtoTree.load(args.prototree).to(device=device)
    # Obtain the dataset and dataloaders
    _, _, _, classes, _ = get_dataloaders(
        dataset=args.dataset,
        projection_mode=None,
        batch_size=args.batch_size,
        disable_cuda=args.disable_cuda,
    )
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])

    sample = test_transform(Image.open(args.sample_dir)).unsqueeze(0).to(device)

    gen_pred_vis(
        tree, sample, args.sample_dir, args.results_dir, args, classes)


if __name__ == '__main__':
    args = get_local_expl_args()
    try:
        Image.open(args.sample_dir)
        print("Image to explain: ", args.sample_dir)
        explain_local(args)
    except:  # folder is not image
        class_name = args.sample_dir.split('/')[-1]
        if not os.path.exists(os.path.join(os.path.join(args.root_dir, args.results_dir), class_name)):
            os.makedirs(os.path.join(os.path.join(args.root_dir, args.results_dir), class_name))
        for filename in os.listdir(args.sample_dir):
            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                args_1 = deepcopy(args)
                args_1.sample_dir = args.sample_dir + "/" + filename
                args_1.results_dir = os.path.join(args.results_dir, class_name)
                explain_local(args_1)
