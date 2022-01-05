#!/usr/bin/env python
# coding: utf-8

import torch
import copy
from tqdm import tqdm
from typing import Callable
from termcolor import colored
from FeatureExtractor import get_encoder
from DataHandling import DataLoaderCyclicIterator


def program_log(msg):
    print(colored(msg, 'blue'))


def minibatch_log(msg):
    print(colored(msg, 'green'))


def matching_accuracy(probabilities: torch.Tensor, real_match_idx=0):
    return (torch.multinomial(input=probabilities, num_samples=1) == real_match_idx).sum() / probabilities.shape[0]


def update_learning_rate(optimizer: torch.optim, epoch: int, total_epochs: int):
    if epoch in [total_epochs * 0.6, total_epochs * 0.8]:
        optimizer.param_groups[0]['lr'] *= 0.1


def get_MoCo_feature_extractor(
        temperature: float,
        loader: torch.utils.data.DataLoader,
        augment: Callable[[torch.Tensor], torch.Tensor],
        momentum: float,
        key_dictionary_size: int,
        num_epochs: int):
    """
    Generates a feature extraction network as described by MoCo v2 paper based on the ResNet50 feature extractor backbone
    :param temperature: hyperparameter defining the density of the contrastive loss function
    :param loader: unlabeled training data loader ################################################################################################# need to UPDATE
    :param augment: augmentation function (random augmentation)
    :param momentum: hyperparameter defining the speed at which the key dictionary is updated
    :param key_dictionary_size: hyperparameter defining the number of keys to maintain.  Should be a   product of the loader batch_size
    :param num_epochs: number of epochs to train the MoCo feature extractor
    :return: feature extraction network
    """

    # f_q, f_k: encoder networks for query and key
    # queue: dictionary as a queue of K keys (CxK)

    # init
    program_log("Initializing feature extractor training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f_q = get_encoder().to(device)
    f_k = copy.deepcopy(f_q).to(
        device)  # create independent copy of f_q that begins with the same parameters but updates more slowly
    optimizer = torch.optim.SGD(f_q.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Generate keys_queue
    program_log("Generating initial keys queue")
    num_initial_key_batches = key_dictionary_size // loader.batch_size
    loader_iterator = DataLoaderCyclicIterator(loader, load_labels=False)

    with torch.no_grad():
        for _ in tqdm(range(num_initial_key_batches)):
            init_k = f_k(augment(next(loader_iterator).to(device))).detach()
            init_k = torch.div(init_k, torch.norm(init_k, dim=1).reshape(-1, 1))
            keys_queue = torch.cat([init_k], dim=0)

    program_log("Beginning training loop")
    f_q.train()
    for epoch in tqdm(range(num_epochs)):
        update_learning_rate(optimizer, epoch, num_epochs)
        total_loss = 0.0
        print(f'epoch = {epoch + 1}')
        for batch, x in enumerate(loader_iterator):  # load a minibatch x with N samples

            optimizer.zero_grad()

            x_q = augment(x).to(device)  # a randomly augmented version
            x_k = augment(x).to(device)  # another randomly augmented version
            q = f_q(x_q).to(device)  # queries: NxC

            with torch.no_grad():  # no gradient to keys
                shuffled_idx = torch.randperm(x_k.size(dim=0))  # permutation of indexes
                shuffled_x_k = x_k[shuffled_idx]  # shuffle keys
                shuffled_k = f_k(shuffled_x_k)  # shuffled keys encoder
                k = torch.zeros_like(shuffled_k)  # restore original keys order
                k[shuffled_idx] = shuffled_k  # keys: NxC

            # normalize query and keys
            q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
            k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

            minibatch_size, sample_size = k.shape
            N, C = minibatch_size, sample_size

            # positive logits: Nx1
            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(dim=2)
            positive_key = torch.exp(torch.div(l_pos, temperature))

            # negative logits: NxK
            l_neg = torch.mm(q.view(N, C), torch.t(keys_queue))
            negative_keys = torch.sum(torch.exp(torch.div(l_neg, temperature)), dim=1)

            denominator = negative_keys + positive_key

            # logits: Nx(1+K)
            logits = torch.cat([positive_key, negative_keys.reshape(-1, 1)], dim=1).to(device)

            loss = torch.mean(-torch.log(torch.div(positive_key, denominator)))
            total_loss += loss.item()

            # SGD update: query network
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for target_param, old_param, new_param in zip(f_k.parameters(), f_k.parameters(), f_q.parameters()):
                    # Edit params in-place
                    target_param.data = momentum * old_param + (1 - momentum) * new_param

            # update dictionary
            keys_queue = torch.cat((keys_queue, k))  # enqueue the current minibatch
            keys_queue = keys_queue[k.shape[0]:]  # dequeue the earliest minibatch

            # print current batch and accuracy results
            if batch % 100 == 0:
                print(f'batch = {batch}')
                minibatch_log(
                    f"Completed minibatch, loss={loss.item()}, with accuracy={matching_accuracy(torch.softmax(logits, dim=1))}")

        print(f'Epoch #{epoch + 1}: Avg. loss={total_loss / len(loader)}')

    program_log("Completed training MoCo feature extractor!")
    return f_q
