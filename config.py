from fastai.basics import *
import torch

# Hyperparams taken from paper
TEMPERATURE = 0.07
MOMENTUM = 0.999
KEY_DICTIONARY_SIZE = 4096
NUM_EPOCHS = 100

path = untar_data(URLs.IMAGENETTE)
CHECKPOINT = 'final_model.pth'
EARLY_STOP = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# out hyperparameters
TRAIN_FEATURES = False
batch_size = 256
lr = 30
weight_decay = 0
momentum = 0.9
