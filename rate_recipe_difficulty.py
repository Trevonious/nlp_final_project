import argparse
import csv
import numpy as np
from matplotlib import pyplot as plt
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os
import glob

from preprocessing.preprocess_data import preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
args = parser.parse_args()
verbose = args.verbose

#################
# General Steps #
#################

# 1. Read in data from relevent csv files
# 2. Split data into training and testing sets
#   - 80% training
#   - 20% testing
# 3. Preprocess data
#   - Tokenize
#   - Remove stop words
#   - Vectorize
# 4. Train model
# 5. Test model
# 6. Output results

##################
# Main Execution #
##################

print()
print('##########################')
print('# Rate Recipe Difficulty #')
print('##########################')
print()

# Get current path
path = os.getcwd()
# Get csv files in data folder
csv_files = glob.glob(os.path.join(path + '\\data', "*.csv"))
# Initialize datasets
interaction_data = []
interaction_data_preprocessed = []
interaction_data_test = []
interaction_data_training = []
recipe_data = []
recipe_data_preprocessed = []
recipe_data_test = []
recipe_data_training = []

# Read in data from csv files
for f in csv_files:
  
  if 'RAW_interactions' in f:
    interaction_data = np.array(pd.read_csv(f))
    # Split data into training and testing sets
    interaction_data_training, interaction_data_test = train_test_split(interaction_data, test_size=0.2)

    if (verbose):
      print('Interaction Training Data size: ' + str(len(interaction_data_training)))
      print('Interaction Testing Data size: ' + str(len(interaction_data_test)))
      print('Interaction Data example: ')
      print(interaction_data_training[0])
      print()
  elif 'RAW_recipes' in f:
    recipe_data = np.array(pd.read_csv(f))
    # Split data into training and testing sets
    recipe_data_training, recipe_data_test = train_test_split(recipe_data, test_size=0.2)

    if (verbose):
      print('Recipe Training Data size: ' + str(len(recipe_data_training)))
      print('Recipe Testing Data size: ' + str(len(recipe_data_test)))
      print('Recipe Data example: ')
      print(recipe_data_training[0])
      print()

# Preprocess data

# Train model

# Test model

# Output results
# show results, graphs, and metrics
