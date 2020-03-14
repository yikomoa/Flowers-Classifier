#coding=UTF-8
import os
import random
from torchvision import models
import numpy as np
import time
import copy
import torch
import input_data
from torch import nn, optim
from torch.optim import lr_scheduler

# ---------------------------------------------------------------------------
# hyperparameters
train_dir = './flower_photos'
ratio = 0.3
SIZE = 224
EPOCHS = 30
seed = 1
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
random.seed(seed) 
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = True
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


model = models.densenet169(pretrained=True)

# for param in model.parameters():
#     param.requires_grad_(False)

classifier = nn.Sequential(
    nn.Linear(1664, 5),
    nn.LogSoftmax(dim=1)
)
model.classifier = classifier

print(model)


dataloaders, dataset_sizes = input_data.get_files(train_dir, ratio, SIZE)


def train_model(model, criteria, optimizer, scheduler, num_epochs, device='cuda'):
    """
    Train the model
    :param model:
    :param criteria:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param device:
    :return:
    """
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
# ------------------------------------------------------------------

criteria = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # 1
sched = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 4
train_model(model, criteria, optimizer, sched, EPOCHS, device)
model_file_name = 'flower_classifier_densenet169_1024.pth'
torch.save({'arch': 'densenet169',
            'state_dict': model.state_dict()},
           model_file_name)