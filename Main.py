import torch
from Datasets import get_loaders
from Augmentations import augment
from MoCoTrainer import get_MoCo_feature_extractor
from tqdm import tqdm


# Hyperparams taken from paper
TEMPERATURE = 0.07
MOMENTUM = 0.999
KEY_DICTIONARY_SIZE = 4096 # FIXME: should be 4096
NUM_EPOCHS = 800


def accuracy(predicted_labels, true_labels):
    return (predicted_labels == true_labels).sum() / len(true_labels)


''' First test MoCoTrainer.py successful train (incomplete), then test quality of extracted features with this classifier
class Classifier(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, loss_func, optimizer_type):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = feature_extractor.to(device=device)
        ftr_exctr_final_layer = self.feature_extractor.fc[-1]
        self.fc = torch.nn.Linear(in_features=ftr_exctr_final_layer.out_features, out_features=ftr_exctr_final_layer.out_features, device=device)
        self.loss_func = loss_func
        self.optimizer = optimizer_type(self.fc.parameters())

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)  # Only fine-tune classifier head
        x = self.fc(x)
        return x

    def train(self, train_loader, val_loader=None):
        validation_accuracies = []
        for _ in tqdm(range(NUM_EPOCHS)):
            for inputs, true_labels in train_loader:
                self.optimizer.zero_grad()

                predicted_labels = self.forward(inputs)
                loss = self.loss_func(predicted_labels, true_labels)
                loss.backward()
                self.optimizer.step()

                print(f"loss={loss}")

                if val_loader:
                    inputs, true_labels = next(val_loader)
                    with torch.no_grad():
                        predicted_labels = self.forward(inputs)
                    validation_accuracies.append(accuracy(predicted_labels=predicted_labels, true_labels=true_labels))

    def test(self, test_loader):
        total_correct = 0
        test_size = 0
        for inputs, true_labels in test_loader:
            predicted_labels = self.forward(inputs)
            total_correct += (predicted_labels == true_labels).sum()
            test_size += len(true_labels)

        print(f"Accuracy: {total_correct / test_size}")
'''

train_loader, test_loader = get_loaders(data_path="imagenette2", batch_size=32)  #FIXME: should be 256

f_q_ = get_MoCo_feature_extractor(temperature=TEMPERATURE, loader=train_loader,
                                  augment=augment, momentum=MOMENTUM,
                                  key_dictionary_size=KEY_DICTIONARY_SIZE, num_epochs=NUM_EPOCHS)
