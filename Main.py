import torch
from Datasets import get_loaders
from Augmentations import augment
from MoCoTrainer import get_MoCo_feature_extractor
from tqdm import tqdm
import fastai
from fastai.vision import *
from fastai.basics import *
from Linear_classifier import Classifier
from Classifier_trainer import ClassifierTrainer
from Plot_results import plot_fit
from FeatureExtractor import get_encoder


# Hyperparams taken from paper
TEMPERATURE = 0.07
MOMENTUM = 0.999
KEY_DICTIONARY_SIZE = 4096
NUM_EPOCHS = 1000

path = untar_data(URLs.IMAGENETTE)
CHECKPOINT = 'final_model.pth'
EARLY_STOP = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

# train_loader, test_loader = get_loaders(data_path='Imagenette2', batch_size=32)
train_loader, test_loader = get_loaders(data_path=path, batch_size=256)
#
# f_q = get_MoCo_feature_extractor(temperature=TEMPERATURE, loader=train_loader,
#                                   augment=augment, momentum=MOMENTUM,
#                                   key_dictionary_size=KEY_DICTIONARY_SIZE, num_epochs=NUM_EPOCHS)

f_q = get_encoder().to(device)
f_q.load_state_dict(torch.load('f_q_weights.pth', map_location=device))
new_f_q = torch.nn.Sequential(*(list(f_q.children())[:-1]))

num_classes_ = len(train_loader.dataset.class_to_idx)

imagenette_classifier = Classifier(feature_extractor=new_f_q, num_classes=num_classes_).to(device)

optimizer = torch.optim.SGD(imagenette_classifier.parameters(), lr=30, weight_decay=0, momentum=0.9)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

trainer = ClassifierTrainer(imagenette_classifier, loss_fn, optimizer)

fit_result = trainer.fit(train_loader, test_loader, num_epochs=NUM_EPOCHS, checkpoints=CHECKPOINT,
                         early_stopping=EARLY_STOP)

fig, axes = plot_fit(fit_result)

fig.savefig('Fit_results.png')

print('finished')
