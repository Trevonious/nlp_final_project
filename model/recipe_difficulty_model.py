import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from analysis.classification_analysis import analyzeRecipeDifficultyClassification
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(7, 5)
    self.fc2 = nn.Linear(5, 2)

  def forward(self, x):
    # Utilize relu activation function
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)

    return x

kFolds = 5
batch_size = 7
num_epochs = 20
learning_rate = 0.01
decay_rate = learning_rate / num_epochs
momentum = 0.8
shuffle = True

def trainModel(recipe_data, verbose):
  current_fold = 1
  # Skip ids column and convert data arrays to tensors
  data = torch.from_numpy(recipe_data)
  ids = recipe_data[:,0]
  # Define the KFold cross-validator
  kf = KFold(n_splits=kFolds, shuffle=shuffle)

  # Start the KFold cross-validation
  for training_index, test_index in kf.split(data):
    outputs = []
    # Split the data
    training_data, test_data = data[training_index], data[test_index]
    test_ids = ids[test_index]

    if verbose:
      print('Beginning training neural network - Fold ' + str(current_fold) + '...')
      print('Verifying training and test data...')
      print('Training data length: ' + str(len(training_data)))
      print('Test data length: ' + str(len(test_data)))
    
    # Create the data loaders
    training_data_loader = DataLoader(TensorDataset(training_data, training_data), batch_size=batch_size)
    # Create the network
    net = Net()
    # Define the loss function
    lossFunction = nn.CrossEntropyLoss()
    # Define the stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay_rate)

    # Train the network
    for epoch in range(num_epochs):
      output = None

      for inputs, original_inputs in training_data_loader:
        # Zero out the parameter gradients
        optimizer.zero_grad()
        # Forward
        output = net(inputs[:,1:8])
        target = original_inputs[:,1]
        # Backward
        loss = lossFunction(output, target.long())
        loss.backward()
        # Optimize
        optimizer.step()

      outputs.append(output)

    current_fold += 1

    print('Training neural network complete!')
    print()

    testModel(test_data, test_ids, net, verbose)

def testModel(test_data, test_ids, trained_network, verbose):
  recipe_difficulties = []
  predicted_labels = []
  test_data_loader = DataLoader(TensorDataset(test_data, test_data), batch_size=batch_size)
  test_ids = []

  print('Testing neural network...')

  with torch.no_grad():

    for inputs, original_inputs in test_data_loader:
      test_ids.append(original_inputs[:,0])
      outputs = trained_network(inputs[:,1:8])
      # Predict the labels
      predictions = torch.mean(outputs.data, dim=1)
      predicted_labels.append(predictions)
    
    recipe_difficulties = analyzeRecipeDifficultyClassification(predicted_labels)

  if verbose:
    print('Test ids length: ' + str(len(test_ids)))
    print('Recipe difficulties length: ' + str(len(recipe_difficulties)))
    
  print('Testing neural network complete!')
  print()

  x = np.arange(len(test_ids))
  y = np.arange(0, 3)
  plt.title('Recipe Difficulty Predictions')
  plt.xlabel('Recipes')
  plt.ylabel('Difficulty')
  plt.xticks(x, test_ids, rotation='vertical')
  plt.yticks(y, ['Easy', 'Medium', 'Hard'])
  plt.plot(x, recipe_difficulties, **{ 'marker': 'o', 'linestyle': 'None' })
  plt.show()
