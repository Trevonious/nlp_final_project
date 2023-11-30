import numpy as np
from sklearn import preprocessing as preproc

def analyzeRecipeDifficultyClassification(classifications):
  easy = 1 / 3
  label_results = []
  medium = 2 / 3

  print('Classifying recipe difficulties...')

  for classification in classifications:
    labels = []
    # Normalize classification to be between 0 and 1, with 1 representing the most difficult
    normalized_classification = preproc.MinMaxScaler().fit_transform(classification.numpy().reshape(-1, 1))
    average = float(np.mean(normalized_classification))

    if average <= easy:
      labels.append(0)
    elif average <= medium:
      labels.append(1)
    else:
      labels.append(2)

    label_results.append(labels)

  return label_results
