from typing import Callable, Any
import torch
from Train_results import BatchResult, EpochResult, FitResult
import torch.nn as nn
from Linear_classifier import Classifier
from torch.utils.data import DataLoader
import tqdm
import sys
import os


class ClassifierTrainer():
    """
    Trainer for our Classifier-based model.
    """

    def __init__(self, model: Classifier, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, num_epochs: int) -> EpochResult:
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_acc = None

        for epoch in range(num_epochs):
            print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")

            # TODO: Train & evaluate for one epoch
            #  - Save losses and accuracies in the lists above.
            train_result = self.train_epoch(dl_train)
            # print(train_result.accuracy)
            train_loss.append(sum(train_result.losses) / len(train_result.losses))
            train_acc.append(train_result.accuracy)

            test_result = self.test_epoch(dl_test)
            test_loss.append(sum(test_result.losses) / len(test_result.losses))
            test_acc.append(test_result.accuracy)

            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch)

    def test_epoch(self, dl_test: DataLoader) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch)

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable[[Any], BatchResult], verbose=True,
                       max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        # print(num_batches)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            # for batch_idx, data in enumerate(dl):
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res[0]:.3f})")
                pbar.update()

                losses.append(batch_res[0])
                num_correct += batch_res[1]

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses, accuracy)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        train_batch_loss: float
        train_batch_num_correct: int

        # train
        y_pred = self.model(X)

        loss = self.loss_fn(y_pred, y)
        train_batch_loss = float(loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        prediction = torch.argmax(y_pred, dim=1)
        train_batch_num_correct = int((y == prediction).detach().sum())

        return BatchResult(train_batch_loss, train_batch_num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        test_batch_loss: float
        test_batch_num_correct: int

        with torch.no_grad():
            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            test_batch_loss = float(loss.item())

            prediction = torch.argmax(y_pred, dim=1)
            test_batch_num_correct = int((y == prediction).detach().sum())

        return BatchResult(test_batch_loss, test_batch_num_correct)
