# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:12:06 2022

@author: harsh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as opt
import tqdm

# Global variables
input_shape = 124
batch_size = 64
input_channels = 3
learning_rate = 0.008
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
num_classes = 2

class FaceMaskClassifier2(nn.Module):
    def __init__(self):
        super(FaceMaskClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=256)
        self.drop = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)


        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=31, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.drop(x)

        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.drop(x)


        x = F.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.drop(x)
    
        x = self.maxpool(x)
        x = self.lastcnn(x)

        return x

# load the dataset
def load_dataset():
    # defining data transformations
    my_transform = transforms.Compose([
         transforms.Resize((input_shape, input_shape)),
         transforms.ColorJitter(brightness=0.4),
         transforms.RandomRotation(degrees=45),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.0, std=1.0)
    ])

    # loading the data
    dataset = datasets.ImageFolder(drive_dataset_path,transform=my_transform)

    # splitting the data into train test and validation
    data_len = len(dataset)
    print("Length of the data is : "+str(data_len))
    train_set_size = int(data_len * 0.8)
    valid_set_size = int(train_set_size * 0.1)+1
    train_set_size2 = int(train_set_size * 0.9)
    test_set_size = int((data_len - train_set_size))
    print(f" Training data size: {train_set_size2}\n Validation data size: {valid_set_size} \n Testing data size: {test_set_size}\n")
    train_data, test_data = data.random_split(dataset, [train_set_size, test_set_size])
    train_data, valid_data = data.random_split(train_data, [train_set_size2, valid_set_size])

    # defining the data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    img, l = next(iter(train_loader))
    return train_loader, valid_loader, test_loader


# Train model

def train2(train_loader, valid_loader, model, criterion, optimizer):
    print("Begin training.")
    for ep in range(num_epochs):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = binary_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in valid_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                #y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = binary_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(valid_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(valid_loader))
        print(
            f'Epoch {ep + 0:02}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(valid_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(valid_loader):.3f}')

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

def evaluate(loader, model, model_type="nn"):
    correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if (model_type == "nn"):
                x = x.reshape(x.shape[0], -1)

            score = model(x)

            y_pred_tag = torch.log_softmax(score, dim=1)
            _, y_pred_tags = torch.max(y_pred_tag, dim=1)
            #_, preds = score.max(1)
            correct += (y_pred_tags == y).sum()
            num_samples += y_pred_tag.shape[0]

    num_samples = 1 if num_samples == 0 else num_samples
    accuracy = 100 * (correct / num_samples)
    return accuracy


def main():
    train_loader, valid_loader, test_loader = load_dataset()

    # Initializing the model

    model = FaceMaskClassifier2().to(device)

    optimizer = opt.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(torch.cuda.is_available())
    print("Model detials: \n")
    print(model)

    print("\n Training The Model")
    train2(train_loader, valid_loader, model, criterion, optimizer)

    return model
