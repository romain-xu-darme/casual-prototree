from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from prototree.prototree import ProtoTree


def train_epoch(
        tree: ProtoTree,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        disable_derivative_free_leaf_optim: bool,
        device: str,
        particul_ratio: float = 0,
        particul_loss: nn.Module = None,
        progress_prefix: str = 'Train Epoch',
) -> dict:

    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_cce_loss = 0.
    total_acc = 0.

    # Init Particul loss function just in case
    total_particul_loss = 0
    total_loc_loss = 0
    total_unq_loss = 0
    total_cls_loss = 0

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        tree_output = tree.forward(xs)
        ys_pred, info = tree_output[0], tree_output[1]

        # Learn prototypes and network with gradient descent.
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        if tree._log_probabilities:
            cce_loss = F.nll_loss(ys_pred, ys)
        else:
            cce_loss = F.nll_loss(torch.log(ys_pred), ys)

        if particul_loss:
            # Add Particul loss
            amaps = tree_output[2]
            part_loss, metrics = particul_loss(None, amaps)
            # Weighted sum of classification loss and Particul loss
            loss = cce_loss*(1-particul_ratio)+particul_ratio*part_loss
            total_particul_loss += metrics[0]
            total_loc_loss += metrics[1]
            total_unq_loss += metrics[2]
            total_cls_loss += metrics[3]
        else:
            loss = cce_loss

        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        if not disable_derivative_free_leaf_optim:
            # Update leaves with derivate-free algorithm
            # Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys]  # shape (batchsize, num_classes)
                for leaf in tree.leaves:
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(
                            torch.logsumexp(info['pa_tensor'][leaf.index]
                                            + leaf.distribution()
                                            + torch.log(target)
                                            - ys_pred,
                                            dim=0))
                    else:
                        update = torch.sum(
                            (info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred,
                            dim=0)
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    # dist_params values can get slightly negative because of floating point issues.
                    # therefore, set to zero.
                    F.relu_(leaf._dist_params)
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        postfix_str = f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, ' \
                      f'CCE loss: {cce_loss.item():.3f}, Acc: {acc:.3f}'
        if particul_loss:
            postfix_str += f' Particul loss: {metrics[0]:.3f} ' \
                           f'(Loc: {metrics[1]:.3f}, Unq: {metrics[2]:.3f}, Cls: {metrics[3]:.3f})'
        train_iter.set_postfix_str(postfix_str)
        # Compute metrics over this batch
        total_loss += loss.item()
        total_cce_loss += cce_loss.item()
        total_acc += acc

    train_info['loss'] = total_loss/nr_batches
    train_info['cce_loss'] = total_cce_loss/nr_batches
    train_info['train_accuracy'] = total_acc/nr_batches
    if particul_loss:
        train_info['particul_loss'] = total_particul_loss/nr_batches
        train_info['loc_loss'] = total_loc_loss/nr_batches
        train_info['unq_loss'] = total_unq_loss/nr_batches
        train_info['cls_loss'] = total_cls_loss/nr_batches
    return train_info


def train_epoch_kontschieder(
        tree: ProtoTree,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        disable_derivative_free_leaf_optim: bool,
        device: str,
        particul_ratio: float = 0,
        particul_loss: nn.Module = None,
        progress_prefix: str = 'Train Epoch',
) -> dict:

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_cce_loss = 0.
    total_acc = 0.
    nr_batches = float(len(train_loader))

    # Init Particul loss function just in case
    total_particul_loss = 0
    total_loc_loss = 0
    total_unq_loss = 0
    total_cls_loss = 0

    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
    else:
        if tree._kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network.
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        tree_output = tree.forward(xs)
        ys_pred = tree_output[0]

        # Compute the loss
        if tree._log_probabilities:
            cce_loss = F.nll_loss(ys_pred, ys)
        else:
            cce_loss = F.nll_loss(torch.log(ys_pred), ys)

        if particul_loss:
            # Add Particul loss
            amaps = tree_output[2]
            part_loss, metrics = particul_loss(None, amaps)
            # Weighted sum of classification loss and Particul loss
            loss = cce_loss*(1-particul_ratio)+particul_ratio*part_loss
            total_particul_loss += metrics[0]
            total_loc_loss += metrics[1]
            total_unq_loss += metrics[2]
            total_cls_loss += metrics[3]
        else:
            loss = cce_loss

        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        postfix_str = f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, ' \
                      f'CCE loss: {cce_loss.item():.3f}, Acc: {acc:.3f}'
        if particul_loss:
            postfix_str += f' Particul loss: {metrics[0]:.3f} ' \
                           f'(Loc: {metrics[1]:.3f}, Unq: {metrics[2]:.3f}, Cls: {metrics[3]:.3f})'
        train_iter.set_postfix_str(postfix_str)
        # Compute metrics over this batch
        total_loss += loss.item()
        total_cce_loss += cce_loss.item()
        total_acc += acc

    train_info['loss'] = total_loss/nr_batches
    train_info['cce_loss'] = total_cce_loss/nr_batches
    train_info['train_accuracy'] = total_acc/nr_batches
    if particul_loss:
        train_info['particul_loss'] = total_particul_loss/nr_batches
        train_info['loc_loss'] = total_loc_loss/nr_batches
        train_info['unq_loss'] = total_unq_loss/nr_batches
        train_info['cls_loss'] = total_cls_loss/nr_batches
    return train_info


# Updates leaves with derivative-free algorithm
def train_leaves_epoch(
        tree: ProtoTree,
        train_loader: DataLoader,
        epoch: int,
        device: str,
        progress_prefix: str = 'Train Leafs Epoch',
) -> None:
    # Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

        # Show progress on progress bar
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)

        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf._dist_params)

        for i, (xs, ys) in train_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Train leafs without gradient descent
            out, info = tree.forward(xs)
            target = eye[ys]  # shape (batchsize, num_classes)
            for leaf in tree.leaves:
                if tree._log_probabilities:
                    # log version
                    update = torch.exp(
                        torch.logsumexp(
                            info['pa_tensor'][leaf.index]
                            + leaf.distribution()
                            + torch.log(target)
                            - out,
                            dim=0))
                else:
                    update = torch.sum(
                        (info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out,
                        dim=0)
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf._dist_params -= leaf._dist_params  # set current dist params to zero
            leaf._dist_params += update_sum[leaf]  # give dist params new value
