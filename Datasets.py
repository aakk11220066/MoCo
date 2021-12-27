import torch
from torchvision import datasets, transforms


def get_loader(data_path, batch_size):
    """

    :param data_path: path to the directory in which we have the train and validation files
    :param batch_size: the size of the train and validation batch
    :return: train and validation images as Torch DataLoaders
    """
    transform_ = {"train": transforms.Compose([transforms.ToTensor()]),
                  "val": transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])}

    ds_train = datasets.ImageFolder(data_path/'train', transform=transform_["train"])
    ds_val = datasets.ImageFolder(data_path/'val', transform=transform_["val"])

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, shuffle=False)

    return dl_train, dl_val
