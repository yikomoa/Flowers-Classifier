#coding=UTF-8
import os
import numpy as np
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_files(file_dir, ratio, SIZE):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(SIZE + 32),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    roses = []
    label_roses = []
    tulips = []
    label_tulips = []
    dandelion = []
    label_dandelion = []
    sunflowers = []
    label_sunflowers = []
    daisy = []
    label_daisy = []

    for file in os.listdir(file_dir + '/roses'):
        roses.append(file_dir + '/roses' + '/' + file)
        label_roses.append(0)
    for file in os.listdir(file_dir + '/tulips'):
        tulips.append(file_dir + '/tulips' + '/' + file)
        label_tulips.append(1)
    for file in os.listdir(file_dir + '/dandelion'):
        dandelion.append(file_dir + '/dandelion' + '/' + file)
        label_dandelion.append(2)
    for file in os.listdir(file_dir + '/sunflowers'):
        sunflowers.append(file_dir + '/sunflowers' + '/' + file)
        label_sunflowers.append(3)
    for file in os.listdir(file_dir + '/daisy'):
        sunflowers.append(file_dir + '/daisy' + '/' + file)
        label_sunflowers.append(4)

    image_list = np.hstack((roses, tulips, dandelion, sunflowers, daisy))
    label_list = np.hstack((label_roses, label_tulips, label_dandelion, label_sunflowers, label_daisy))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio)) 
    n_train = n_sample - n_val 
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    fd1 = FlowerDataset(tra_images, tra_labels, data_transforms['train'])
    fd2 = FlowerDataset(val_images, val_labels, data_transforms['valid'])
    dataloaders = {'train': DataLoader(fd1, 32, True), 'valid': DataLoader(fd2, 32, True)}
    dataset_sizes = {'train': len(fd1), 'valid': len(fd2)}
    return dataloaders, dataset_sizes


class FlowerDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        path = self.inputs[index]
        label = self.labels[index]
        img = self.__loader(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

    def __loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


