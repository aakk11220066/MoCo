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
    :param loader: unlabeled training data loader
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
    optimizer = torch.optim.SGD(f_q.parameters(), lr=0.03, weight_decay=1e-4, momentum=0.9)
    f_k = copy.deepcopy(f_q)  # create independent copy of f_q that begins with the same parameters but updates more slowly
    loss_fn = torch.nn.CrossEntropyLoss()

    # Generate keys_queue
    program_log("Generating initial keys queue")
    num_initial_key_batches = key_dictionary_size // loader.batch_size
    loader_iterator = DataLoaderCyclicIterator(loader, load_labels=False)
    with torch.no_grad():
        keys_queue = torch.cat([
            f_k(
                augment(next(loader_iterator))
            )
            for _ in tqdm(range(num_initial_key_batches))
        ], dim=0)

    program_log("Beginning training loop")
    for epoch in tqdm(range(num_epochs)):
        update_learning_rate(optimizer, epoch, num_epochs)
        for x in loader_iterator:  # load a minibatch x with N samples
            optimizer.zero_grad()

            x_q = augment(x)  # a randomly augmented version
            x_k = augment(x)  # another randomly augmented version
            q = f_q(x_q)  # queries: NxC
            with torch.no_grad():  # no gradient to keys
                k = f_k(x_k)  # keys: NxC
            minibatch_size, sample_size = k.shape
            N, C, K = minibatch_size, sample_size, key_dictionary_size

            # positive logits: Nx1
            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(dim=2)
            # negative logits: NxK
            l_neg = torch.mm(q.view(N, C), keys_queue.view(C, K))
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # contrastive loss, Eqn.(1)
            labels = torch.zeros((N,), dtype=torch.long)  # positives are the 0-th
            loss = loss_fn(logits/temperature, labels)

            # SGD update: query network
            loss.backward()
            optimizer.step()

            # momentum update: key network
            with torch.no_grad():
                for target_param, old_param, new_param in zip(f_k.parameters(), f_k.parameters(), f_q.parameters()):
                    # Edit params in-place
                    target_param.data = momentum*old_param + (1-momentum)*new_param

            # update dictionary
            keys_queue = torch.cat((keys_queue, k))  # enqueue the current minibatch
            keys_queue = keys_queue[k.shape[0]:]  # dequeue the earliest minibatch

            minibatch_log(f"Completed minibatch, loss={loss.item()}, with accuracy={matching_accuracy(torch.softmax(logits, dim=1))}")

    program_log("Completed training MoCo feature extractor!")
    return f_q
