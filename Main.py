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
import config as conf


train_loader, test_loader = get_loaders(data_path='Imagenette2', batch_size=256)
# train_loader, test_loader = get_loaders(data_path=conf.path, batch_size=conf.batch_size)

if conf.TRAIN_FEATURES:
    f_q = get_MoCo_feature_extractor(temperature=conf.TEMPERATURE, loader=train_loader,
                                      augment=augment, momentum=conf.MOMENTUM,
                                      key_dictionary_size=conf.KEY_DICTIONARY_SIZE, num_epochs=conf.NUM_EPOCHS)
else:
    f_q = get_encoder().to(conf.device)
    f_q.load_state_dict(torch.load('f_q_weights.pth', map_location=conf.device))
new_f_q = torch.nn.Sequential(*(list(f_q.children())[:-1]))

num_classes_ = len(train_loader.dataset.class_to_idx)

imagenette_classifier = Classifier(feature_extractor=new_f_q, num_classes=num_classes_).to(conf.device)

optimizer = torch.optim.SGD(imagenette_classifier.parameters(), lr=conf.lr, weight_decay=conf.weight_decay,
                            momentum=conf.momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

loss_fn = torch.nn.CrossEntropyLoss().to(conf.device)

trainer = ClassifierTrainer(imagenette_classifier, loss_fn, optimizer, scheduler)

fit_result = trainer.fit(train_loader, test_loader, num_epochs=conf.NUM_EPOCHS, checkpoints=conf.CHECKPOINT,
                         early_stopping=conf.EARLY_STOP)

fig, axes = plot_fit(fit_result)

fig.savefig(f'Fit_results_{fit_result.num_epochs}.png')

print('finished')
