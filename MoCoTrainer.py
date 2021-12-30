#!/usr/bin/env python
# coding: utf-8

import torch
import copy
from tqdm import tqdm
from typing import Callable
from termcolor import colored
from FeatureExtractor import get_encoder
from DataHandling import DataLoaderCyclicIterator


def log(msg):
    print(colored(msg, 'blue'))


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
    log("Initializing feature extractor training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f_q = get_encoder().to(device)
    optimizer = torch.optim.SGD(f_q.parameters(), lr=0.03, weight_decay=1e-4, momentum=0.9)
    f_k = copy.deepcopy(f_q)  # create independent copy of f_q that begins with the same parameters but updates more slowly

    # Generate keys_queue
    log("Generating initial keys queue")
    num_initial_key_batches = key_dictionary_size // loader.batch_size
    loader_iterator = DataLoaderCyclicIterator(loader)
    keys_queue = torch.stack([
        f_k(
            augment(next(loader_iterator))
        )
        for _ in tqdm(range(num_initial_key_batches))
    ])

    log("Beginning training loop")
    for epoch in tqdm(range(num_epochs)):
        update_learning_rate(optimizer, epoch, num_epochs)
        for x in loader_iterator:  # load a minibatch x with N samples
            x_q = augment(x)  # a randomly augmented version
            x_k = augment(x)  # another randomly augmented version
            q = f_q(x_q)  # queries: NxC
            k = f_k(x_k)  # keys: NxC
            k = k.detach()  # no gradient to keys
            minibatch_size, sample_size = k.shape
            N, C, K = minibatch_size, sample_size, key_dictionary_size

            # positive logits: Nx1
            l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
            # negative logits: NxK
            l_neg = torch.mm(q.view(N, C), keys_queue.view(C, K))
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=0)
            print(f"logits shape = {logits.shape}")

            # contrastive loss, Eqn.(1)
            labels = torch.zeros_like(logits)  # positives are the 0-th
            loss = torch.nn.CrossEntropyLoss(logits/temperature, labels)

            # SGD update: query network
            loss.backward()
            optimizer.step()
            # momentum update: key network
            f_k.params = momentum*f_k.parameters() + (1-momentum)*f_q.parameters()  # FIXME: f_k.parameters()

            # update dictionary
            keys_queue = torch.cat((keys_queue, k))  # enqueue the current minibatch
            keys_queue = keys_queue[k.shape[0]:]  # dequeue the earliest minibatch

    log("Completed training MoCo feature extractor!")
    return f_q
