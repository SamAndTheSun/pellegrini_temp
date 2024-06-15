import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import pandas as pd
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display

'''
Functions utilized in the death prediction model
'''

class Model(nn.Module):
  def __init__(self, n_inputs=9, h1=70, h2=70, out_features=1):
    super().__init__()
    self.fc1 = nn.Linear(n_inputs, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
  
  '''
  Things I've tried to improve the model which have not worked:

    Batch normalization
    Dropout
    Autoencoder for noise reduction

  Likely, these don't work due to not having much data and due to the complexity of the problem involved
  '''

def time_to_death_grouped(data, category):
    '''
    groups predicted time until death value by selected category

      param data: inputted dataframe, includes predicted time until death and category columns, df
      param category: the category, i.e. column name, to group dataframe by, str

      return: dataframe grouped by category, df
    '''
    print(f'Average time to death estimate by {category}:\n')
    grouped_data = data.groupby([category, 'Predicted time until death'], as_index=False).mean(numeric_only=True).groupby(category)['Predicted time until death'].mean(numeric_only=True)
    grouped_data = pd.DataFrame(grouped_data)
    grouped_data = grouped_data.sort_values(by='Predicted time until death', ascending=False)
    print(grouped_data)
    print('\n')
    return grouped_data

def cross_validation(X_all, y_all, epochs, batch_size, n_iterations):
    '''
    runs cross validation to determine loss of neural network model

      param X_all: all expected input values, df
      param y_all: all expected output values, df
      param epochs: number of times to process a given batch, int
      param batch_size: size of each batch to be run by each iteration of the NN, int
      param n_iterations: number of entries within cross validation, int

      return: 3 lists containing all of the models predictions, the actual values, and the loss values
    '''

    all_approx = []
    all_actual = []
    all_losses = []

    progress_bar = IntProgress(min=0, max=n_iterations, description=f'Training')
    display(progress_bar)

    n = 0
    while n < n_iterations:

        s = X_all.shape[0]//n_iterations #split into equal parts

        start = s*n
        end = (s*(n+1))+1

        X_val = X_all[start:end]
        y_val = y_all[start:end]

        X_train_1 = X_all[0:start]
        X_train_2 = X_all[end:]
        X_train = pd.concat([X_train_1, X_train_2], axis=0, ignore_index=True)

        y_train_1 = y_all[0:start]
        y_train_2 = y_all[end:]
        y_train = pd.concat([y_train_1, y_train_2], axis=0, ignore_index=True)

        model = train_nn(X_train, y_train, epochs, batch_size, bar=False)
        losses, approx, actual = test_nn(model, X_val, y_val)

        for a in approx:
            if a == None:
                break
            else:
                all_approx.append(a)
        
        for a in actual:
            if a == None:
                break
            else:
                all_actual.append(a)

        for l in losses:
            if l == None:
                break
            else:
                all_losses.append(l)

        n+=1

        progress_bar.value = n

    return all_approx, all_actual, all_losses

def train_nn(X_train, y_train, batch_size, epochs, bar=False):
  '''
  trains neural network model

    param X_train: training set input, df
    param y_train: training set ouput, df
    param batch_size: batch size for neural network, i.e. number of input/output pairs computed at once, int
    param epochs: number of times to sample and process a batch, int

    return: most optimal neural network model identified throughout training, pytorch object
  
  '''

  n_inputs = X_train.shape[1]

  X_train = X_train.values
  y_train = y_train.values

  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

  criterion = nn.SmoothL1Loss()
  dataset = TensorDataset(X_train, y_train)
  dataloader = DataLoader(dataset, batch_size=batch_size)
    
  while True:

    model = Model(n_inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    last_loss = 75
    best_loss = last_loss
    bad_iter = False

    if bar:
      progress_bar = IntProgress(min=0, max=epochs, description='Training')
      display(progress_bar)

    for i in range(epochs):
      try:

        for inputs, targets in dataloader:

          outputs = model(inputs)
          loss = criterion(outputs, targets)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # print every 20 epochs
        if (i % 20) == 0:
          if loss >= last_loss and loss >= 50:
            print('\nAberrant training detected, retrying iteration\n')
            progress_bar.value = 0
            raise Exception

        
        last_loss = loss
        if loss < best_loss:
           best_loss = loss
           best_model = copy.deepcopy(model)

        if bar: progress_bar.value = epochs+1

      except:
        bad_iter = True 
        break

    if bad_iter == False:
      return best_model
    
    else:
      pass

def test_nn(model, X_val, y_val):
  '''
  evalues performance of neural network model
  
    param model: trained NN, pytorch object
    param X_val: input values, df
    param y_val: expected output values: df

    return: 3 lists containing the models predictions, the actual values, and the loss values
  '''

  model.eval()

  X_val = X_val.values
  y_val = y_val.values

  X_val = torch.tensor(X_val, dtype=torch.float32)
  y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

  dataset = TensorDataset(X_val, y_val)
  criterion = nn.SmoothL1Loss()
  dataloader = DataLoader(dataset)

  approx = []
  actual = []
  losses = []

  for inputs, targets in dataloader:

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(float(loss.detach().numpy()))

    pred = outputs.detach().numpy()[0]
    pred = str(pred)
    approx.append(pred[1:-1])

    act = targets.detach().numpy()[0]
    act = str(act)
    actual.append(act[1:-1])

  return losses, approx, actual

def generate_nn_pred(model, X):
  '''
  gets predictions for input values using trained NN

    param model: trained NN, torch object
    param X: input values, df

    returns: n predictions, numpy array
  '''

  model.eval()

  X = X.values
  X = torch.tensor(X, dtype=torch.float32)

  outputs = model(X)
  outputs = outputs.detach().numpy() 

  return outputs

class AutoEncoder(nn.Module):
  def __init__(self, n_inputs=9, h1=6, out_features=3):
    super().__init__()
    #encoding functions
    self.fc1 = nn.Linear(n_inputs, h1)
    self.out = nn.Linear(h1, out_features)

    #decoding functions
    self.out_r = nn.Linear(out_features, h1)
    self.fc1_r = nn.Linear(h1, n_inputs)

  def encode(self, x):
    x = F.relu(self.fc1(x)) #takes the input then modifies it
    x = self.out(x) #gives the output prediction
    return x

  def decode(self, x):
    x = F.relu(self.out_r(x)) #takes the output then modifies it
    x = self.fc1_r(x) #gives the input prediction
    return x

  def forward(self, x):
    x = self.encode(x)
    x = self.decode(x)
    return x
  
def train_ae(X_train, h1=6, out_features=3, batch_size=64, epochs=80):
  n_inputs = X_train.shape[1]

  try: X_train = X_train.values
  except: pass

  #add noise to X_train so the model learns to detect noise
  noise = np.random.normal(1, 0.3, X_train.shape) 
  noisy_X_train = X_train + noise

  X_train = torch.tensor(X_train, dtype=torch.float32)
  noisy_X_train = X_train.clone().detach()

  criterion = nn.MSELoss()
  dataset = TensorDataset(noisy_X_train, X_train)
  dataloader = DataLoader(dataset, batch_size=batch_size)

  model = AutoEncoder(n_inputs, h1=6, out_features=3)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  best_loss = 100
  for i in range(epochs):
    for inputs, targets in dataloader:

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if i % 10 == 0:
      print(f'Epoch: {i} and loss: {loss}')

    if loss < best_loss:
      best_loss = loss
      best_model = copy.deepcopy(model)

  #now, utilize the trained model to predict the values
      
  return best_model

def test_ae(model, X_test):

  try: X_test = X_test.values
  except: pass

  X_test = torch.tensor(X_test, dtype=torch.float32)
  model.eval()

  dataset = TensorDataset(X_test, X_test)
  dataloader = DataLoader(dataset)

  denoised_data = []

  with torch.no_grad():
      for inputs, temp in dataloader:
          outputs = model(inputs)
          denoised_data.append(outputs.squeeze().detach().numpy())

  denoised_data = torch.tensor(np.stack(denoised_data, axis=1), dtype=torch.float32)

  return denoised_data
