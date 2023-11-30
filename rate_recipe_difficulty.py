import argparse
import glob
import numpy as np
import os
import pandas as pd
from model.recipe_difficulty_model import trainModel
from preprocessing.preprocess_data import averageRecipeData, preprocessData

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all_data', help='Execute with all interaction data', type=bool, default=False)
parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
args = parser.parse_args()
all_data = args.all_data
verbose = args.verbose

#################
# General Steps #
#################

# 1. Read in data from relevent csv files
# 2. Preprocess data
#   - Consolidate data
#   - Tokenize
#   - Remove stop words
#   - Sentiment analysis
# 3. Split data into training and testing sets
#   - 80% training
#   - 20% testing
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
print('Begin execution...')
print()

# Get current path
path = os.getcwd()
# Get csv files in data folder
csv_files = glob.glob(os.path.join(path + '\\data', "*.csv"))
# Initialize datasets
interaction_data = []
preprocessed_data = []
test_data = []
training_data = []
recipe_data = []

# Read in data from csv files
for f in csv_files:
  
  if 'RAW_interactions' in f:
    interaction_data = np.array(pd.read_csv(f))
  elif 'RAW_recipes' in f:
    recipe_data = np.array(pd.read_csv(f))

# Preprocess data
preprocessed_data = preprocessData(interaction_data, recipe_data, all_data, verbose)

print('Preparing model data...')

ids = np.array([np.array(preprocessed_data[:,0], dtype=np.float32)]).transpose()
ratings = np.array([np.array(preprocessed_data[:,1], dtype=np.float32)]).transpose()
minutes = np.array([np.array(preprocessed_data[:,4], dtype=np.float32)]).transpose()
n_steps = np.array([np.array(preprocessed_data[:,5], dtype=np.float32)]).transpose()
n_ingredients = np.array([np.array(preprocessed_data[:,6], dtype=np.float32)]).transpose()
review_ratings = np.array([np.array(preprocessed_data[:,8], dtype=np.float32)]).transpose()
review_no_stop_words_num_words = np.array([np.array(preprocessed_data[:,9], dtype=np.float32)]).transpose()
review_sentiments = np.array([np.array(preprocessed_data[:,10], dtype=np.float32)]).transpose()
# Combine all preprocessed data into one array; add ids for averaging
preprocessed_model_data = np.concatenate([ids, ratings, minutes, n_steps, n_ingredients, review_ratings, review_no_stop_words_num_words, review_sentiments], axis=1)
preprocessed_model_data = averageRecipeData(preprocessed_model_data, verbose)

if verbose:
  print('Model data shape: ' + str(preprocessed_model_data.shape))
  print('Model data columns:')
  print('\tColumn 0: recipe_id')
  print('\tColumn 1: rating')
  print('\tColumn 2: minutes')
  print('\tColumn 3: n_steps')
  print('\tColumn 4: n_ingredients')
  print('\tColumn 5: final_review_rating')
  print('\tColumn 6: review_no_stop_words_num_words')
  print('\tColumn 7: review_sentiment')

print('Preparing model data complete!')
print()

# Train model
trainModel(preprocessed_model_data, verbose)

print('Execution complete!')
print()
