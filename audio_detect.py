# Code taken/adapted from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

import os
from pyexpat import model 
import pandas as pd
import math, random
import torch
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
from IPython.display import Audio

import data_script as d
import transforms as t
import model as m

train_data = d.train_df.sample(frac=1)
test_data = d.test_df.sample(frac=1)

from torch.utils.data import random_split

train_myds = t.SoundDS(train_data)
test_myds = t.SoundDS(test_data)

# Random split of 80:20 between training and validation
num_items = len(test_myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(test_myds, [num_train, num_val])

# # Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_myds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# Create the model and put it on the GPU if available
myModel = m.AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, val_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []
  for epoch in range(num_epochs):
    print('----------Epoch----------------')
    running_loss = 0.0
    t_running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    train_loss.append(avg_loss)
    train_acc.append(acc)

    correct_prediction = 0
    total_prediction = 0
    for i, data in enumerate(val_dl):
        with torch.no_grad():
        # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)
            t_loss = criterion(outputs, labels)
            t_running_loss += t_loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    num_batches = len(val_dl)
    avg_loss = t_running_loss / num_batches
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    test_acc.append(acc)
    test_loss.append(avg_loss)       

  print('Finished Training')

  x = list(range(num_epochs))
  plt.figure(0)
  plt.plot(x, train_loss, label='training')
  plt.plot(x, test_loss, label='testing')
  plt.legend(loc='best', frameon=False)
  plt.xlabel('Epoch')
  plt.ylabel('Avg. Loss')
  plt.savefig('train_test_curves.png')

  plt.figure(1)
  plt.plot(x, train_acc, label='training')
  plt.plot(x, test_acc, label='testing')
  plt.legend(loc='best', frameon=False)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.savefig('train_test_acc_curves.png')

num_epochs= 10   # Just for demo, adjust this higher.
training(myModel, train_dl, val_dl, num_epochs)


# # # ----------------------------
# # # Inference
# # # ----------------------------
# # def inference (model, val_dl):
# #   correct_prediction = 0
# #   total_prediction = 0

# #   # Disable gradient updates
# #   with torch.no_grad():
# #     for data in val_dl:
# #       # Get the input features and target labels, and put them on the GPU
# #       inputs, labels = data[0].to(device), data[1].to(device)

# #       # Normalize the inputs
# #       inputs_m, inputs_s = inputs.mean(), inputs.std()
# #       inputs = (inputs - inputs_m) / inputs_s

# #       # Get predictions
# #       outputs = model(inputs)

# #       # Get the predicted class with the highest score
# #       _, prediction = torch.max(outputs,1)
# #       # Count of predictions that matched the target label
# #       correct_prediction += (prediction == labels).sum().item()
# #       total_prediction += prediction.shape[0]
    
# #   acc = correct_prediction/total_prediction
# #   print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# # # Run inference on trained model with the validation set
# # inference(myModel, val_dl)