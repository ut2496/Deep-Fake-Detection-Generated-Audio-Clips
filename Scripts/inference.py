import os
from pyexpat import model
import pandas as pd
import math, random
import torch
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
from IPython.display import Audio

import transforms as t
import model as m
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

def testInf(test_df):

  test_data = test_df.sample(frac=1)
  test_myds = t.SoundDS(test_data)

  test_dl = torch.utils.data.DataLoader(test_myds, batch_size=16, shuffle=False, num_workers=4)
  ### Based on the Model being inferred please change the model class here
  myModel = m.AudioClassifier()
  criterion = nn.CrossEntropyLoss()

  saved_model = torch.load('audioModel_All.pt')

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  myModel = myModel.to(device)

  next(myModel.parameters()).device

  myModel.load_state_dict(saved_model)
  myModel.eval()

  running_loss = 0.0
  t_running_loss = 0.0
  correct_prediction = 0
  total_prediction = 0

  for i, data in enumerate(test_dl):
    with torch.no_grad():
      inputs, labels = data[0].to(device), data[1].to(device)
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s
      outputs = myModel(inputs)
      t_loss = criterion(outputs, labels)
      t_running_loss += t_loss.item()
      _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
  acc = correct_prediction/total_prediction
  num_batches = len(test_dl)
  avg_loss = t_running_loss / num_batches
  print(f'Accuracy: {acc:.2f}, Loss: {avg_loss:.2f}, Total items: {total_prediction}')

class_ids = {
    'real': 0,
    'fake': 1
}

data_list = []
data_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/jsut_multi_band_melgan')
for sound in data_entries:
    v_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/jsut_multi_band_melgan/' + sound, 'class': class_ids['fake'] }
    data_list.append(v_data)

data_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/jsut_parallel_wavegan')
for sound in data_entries:
    v_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/jsut_parallel_wavegan/' + sound, 'class': class_ids['fake'] }
    data_list.append(v_data)

data_entries = os.listdir('/scratch/us450/GenAudioProject/jsutData/jsut_ver1.1/basic5000/wav')
for sound in data_entries:
    v_data = {'file path': '/scratch/us450/GenAudioProject/jsutData/jsut_ver1.1/basic5000/wav/' + sound, 'class': class_ids['real'] }
    data_list.append(v_data)
df = pd.DataFrame(data = data_list)
print("JSUT")
testInf(df)
