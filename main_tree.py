from util.args import get_args, save_args, get_optimizer, load_args
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample

import torch
from copy import deepcopy

# Use onyl deterministic algorithms
torch.use_deterministic_algorithms(True)

def run_tree(args=None):
    args = args or get_args()

    resume = False
    if (os.path.exists(args.log_dir) and os.path.exists(args.log_dir+'/metadata')
            and load_args(args.log_dir+'/metadata') == args and os.path.exists(args.log_dir+'/checkpoints/latest')) \
            or args.state_dict_dir_tree != '':
        # Directory already exists and contains the same arguments => resume computation
        # Alternatively, if state_dict_dir_tree is specified, restart from a given checkpoint
        resume = True

    # Create a logger
    log = Log(args.log_dir, mode='a' if resume else 'w')
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc',
                   'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    log.log_message('Device used: '+str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)

    if not resume:
        # Create a convolutional network based on arguments and add 1x1 conv layer
        features_net, add_on_layers = get_network(num_channels, args)
        # Create a ProtoTree
        tree = ProtoTree(num_classes=len(classes),
                         feature_net=features_net,
                         args=args,
                         add_on_layers=add_on_layers)
        tree = tree.to(device=device)
        # Determine which optimizer should be used to update the tree parameters
        optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones,
                                                         gamma=args.gamma)
        tree, epoch = init_tree(tree, optimizer, scheduler, device, args)
        log.log_message(
            "Max depth %s, so %s internal nodes and %s leaves" % (args.depth, tree.num_branches, tree.num_leaves))
        analyse_output_shape(tree, trainloader, log, device)

        leaf_labels = dict()
        best_train_acc = 0.
        best_test_acc = 0.

        save_checkpoint(
            f'{log.checkpoint_dir}/tree_init', tree, optimizer, scheduler, 0,
            best_train_acc, best_test_acc, leaf_labels, args)

    else:
        # Either latest checkpoint or the one pointed by state_dict_dir_tree
        directory_path = log.checkpoint_dir+'/latest' if not args.state_dict_dir_tree else args.state_dict_dir_tree
        print('Resuming computation from ' + directory_path)
        tree, (optimizer, params_to_freeze, params_to_train), scheduler, stats = \
            load_checkpoint(directory_path)
        best_train_acc, best_test_acc, leaf_labels, epoch = stats
        # Go to the next epoch
        epoch += 1

    if epoch < args.epochs+1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s" % str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(tree, epoch, params_to_freeze, params_to_train, args, log)
            log_learning_rates(optimizer, args, log)

            # Train tree
            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(
                    tree, trainloader, optimizer, epoch,
                    args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            else:
                train_info = train_epoch(
                    tree, trainloader, optimizer, epoch,
                    args.disable_derivative_free_leaf_optim, device, log, log_prefix)
            # Update scheduler and leaf labels before saving checkpoints
            scheduler.step()
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)

            # Update best train accuracy (if necessary)
            best_train_acc = save_best_train_tree(
                tree, optimizer, scheduler, epoch,
                train_info['train_accuracy'], best_train_acc, best_test_acc, leaf_labels, args, log)
            save_tree(
                tree, optimizer, scheduler, epoch,
                best_train_acc, best_test_acc, leaf_labels, args, log)

            # Evaluate tree
            if args.epochs <= 100 or epoch % 10 == 0 or epoch == args.epochs:
                eval_info = eval(tree, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_tree(
                    tree, optimizer, scheduler, epoch,
                    best_train_acc, original_test_acc, best_test_acc, leaf_labels, args, log)
                log.log_values('log_epoch_overview', epoch,
                               original_test_acc, train_info['train_accuracy'], train_info['loss'])
            else:
                log.log_values('log_epoch_overview', epoch, "n.a.",
                               train_info['train_accuracy'], train_info['loss'])

    else: #tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        # Readjust epoch index
        epoch = args.epochs
        eval_info = eval(tree, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_tree(
            tree, optimizer, scheduler, epoch,
            best_train_acc, original_test_acc, best_test_acc, leaf_labels, args, log)
        log.log_values('log_epoch_overview', epoch, original_test_acc, "n.a.", "n.a.")

    '''
        EVALUATE AND ANALYSE TRAINED TREE
    '''
    log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n"%(str(best_train_acc), str(best_test_acc)))
    # Save tree (for sanity checks)
    save_checkpoint(f'{log.checkpoint_dir}/trained',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)

    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch+1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        PRUNE
    '''
    prune(tree, args.pruning_threshold_leaves, log)
    name = "pruned"
    save_checkpoint(f'{log.checkpoint_dir}/{name}',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    pruned_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree
    leaf_labels = analyse_leafs(tree, epoch+2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_test_acc = eval_info['test_accuracy']

    '''
        PROJECT
    '''
    project_info, tree = project_with_class_constraints(tree, projectloader, device, args, log)
    name = "pruned_and_projected"
    save_checkpoint(f'{log.checkpoint_dir}/{name}',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    leaf_labels = analyse_leafs(tree, epoch+3, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    eval_info = eval(tree, testloader, name, device, log)
    pruned_projected_test_acc = eval_info['test_accuracy']
    eval_info_samplemax = eval(tree, testloader, name, device, log, 'sample_max')
    get_avg_path_length(tree, eval_info_samplemax, log)
    eval_info_greedy = eval(tree, testloader, name, device, log, 'greedy')
    get_avg_path_length(tree, eval_info_greedy, log)
    fidelity_info = eval_fidelity(tree, testloader, device, log)

    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, projectloader, name, args, log)
    # visualize tree
    gen_vis(tree, name, args, classes)

    return trained_tree.to('cpu'), pruned_tree.to('cpu'), pruned_projected_tree.to('cpu'), \
        original_test_acc, pruned_test_acc, pruned_projected_test_acc, \
        project_info, eval_info_samplemax, eval_info_greedy, fidelity_info


if __name__ == '__main__':
    args = get_args()
    run_tree(args)
