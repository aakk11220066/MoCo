import torch
from torchvision import datasets, transforms
from os.path import join


def get_loaders(data_path, batch_size):
    """

    :param data_path: path to the directory in which we have the train and validation files
    :param batch_size: the size of the train and validation batch
    :return: train and validation images as Torch DataLoaders
    """
    transform_ = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2023, 0.1994, 0.2010])]),
                  "val": transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                  [0.2023, 0.1994, 0.2010])])}

    ds_train = datasets.ImageFolder(join(data_path, 'train'), transform=transform_["train"])
    ds_val = datasets.ImageFolder(join(data_path, 'val'), transform=transform_["val"])

    # Set num_workers = 0 if this causes crashes on your machine (it's dependent on how many subcores you have available)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=0)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, shuffle=True, num_workers=0)

    return dl_train, dl_val
