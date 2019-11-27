
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as f
import os
import torch
import numpy as np
from PIL import Image

LOAD_PATH = 'results/CNNtorch-100e-16bs-cnn.pt'
OG_DATA_PATH = 'data/test/neut/og'
N_DATA_PATH = 'data/test/neut/n'
SAVE_NAME = 'data/test/results'

INPUT_SIZE = 541
ft_size = ((round(INPUT_SIZE/4)) +1) * ((round(INPUT_SIZE/4)) +1) * 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)
        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)
        self.block3 = nn.Linear(20, 20)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, NUM_CLASSES=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(ft_size, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = torch.load(LOAD_PATH)
model.eval()
if torch.cuda.device_count() > 0:
    if torch.cuda.device_count() > 1:
                  print("Let's use", torch.cuda.device_count(), "GPUs!")
                  model = nn.DataParallel(model)
    model= model.to(device)



data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

for DATA_PATH in [OG_DATA_PATH, N_DATA_PATH]:
    X = []
    for im in os.listdir(DATA_PATH):
        img_obj = Image.open(DATA_PATH + '/' + im)
        X.append(img_obj)
    ins_OG = []
    preds_OG = []
    ins_N = []
    preds_N = []
    D = [data_transforms(x) for x in X]
    for d in D:
        input = d.unsqueeze(0).cuda()
        predictions = f.softmax(model(input)).detach().cpu().numpy()
        predictions = np.squeeze(predictions)
        if DATA_PATH == OG_DATA_PATH:
            ins_OG.append(input)
            preds_OG.append(predictions)
        else:
            ins_N.append(input)
            preds_N.append(input)

OGS = zip(ins_OG, preds_OG)
NS = zip(ins_N, preds_N)

with open((SAVE_NAME + '.csv'), 'w') as csvfile:
        s = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        s.writerow(['images:', OGS])
        s.writerow(['outputs', NS]),












#
