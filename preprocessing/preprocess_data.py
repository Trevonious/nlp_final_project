import numpy as np
import re
from analysis.sentiment_analysis import analyzeReviewSentiment
from analysis.token_analysis import analyzeReviewRatings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

notPunctuation = re.compile('.*[A-Za-z0-9].*')

def preprocessData(interactions, recipes, all_data, verbose):

  print('Begin preprocessing data...')

  recipe_data = consolidateData(interactions, recipes, all_data, verbose)
  review_data = recipe_data[:,2]

  print('Tokenizing reviews...')

  # Tokenize the review data
  tokenized_reviews = []

  for review in review_data:
    tokenized_reviews.append(word_tokenize(str(review)))

  print('Removing stop words from tokenized reviews...')

  # Remove stop words
  reviews_no_stop_words = []
  stop_words = set(stopwords.words('english'))
  token_dictionary = {}
  tokenized_reviews_no_stop_words = []


  for review in tokenized_reviews:
    review_tokens = []

    for token in review:

      # Do not add punctuations or stop words
      if token not in stop_words and notPunctuation.match(token):

        if token not in token_dictionary:
          token_dictionary[token] = 1
        else:
          token_dictionary[token] += 1

        review_tokens.append(token)

    reviews_no_stop_words.append(str(review_tokens))
    tokenized_reviews_no_stop_words.append(review_tokens)

  reviews_no_stop_words = np.array([reviews_no_stop_words]).transpose()
  preprocessed_data = analyzeReviewRatings(tokenized_reviews_no_stop_words, token_dictionary, recipe_data, reviews_no_stop_words, verbose)
  # Sentiment analysis
  preprocessed_data = analyzeReviewSentiment(preprocessed_data, verbose)

  if verbose:
    print('Preprocessed data shape: ' + str(preprocessed_data.shape))
    print('Preprocessed data columns:')
    print('\tColumn 0: recipe_id')
    print('\tColumn 1: rating')
    print('\tColumn 2: review')
    print('\tColumn 3: name')
    print('\tColumn 4: minutes')
    print('\tColumn 5: n_steps')
    print('\tColumn 6: n_ingredients')
    print('\tColumn 7: reviews_no_stop_words')
    print('\tColumn 8: final_review_rating')
    print('\tColumn 9: review_no_stop_words_num_words')
    print('\tColumn 10: review_sentiment')

  print('Preprocessing data complete!')
  print()

  return preprocessed_data

def consolidateData(interactions, recipe_data, all_data, verbose):
  interaction_data = np.array(interactions)
  recipe_data = np.array(recipe_data)

  print('Removing unnecessary columns...')

  # Remove unnecessary columns: user_id (0), date (2)
  interaction_data = np.delete(interaction_data, [0,2], 1)
  # Remove unnecessary columns: contributor (3), submitted (4), tags (5), nutrition (6), steps (8), description (9), ingredients (10)
  recipe_data = np.delete(recipe_data, [3,4,5,6,8,9,10], 1)

  if verbose:
    print('Interaction data shape: ' + str(interaction_data.shape))
    print('Recipe data shape: ' + str(recipe_data.shape))

  print('Consolidating and normalizing data...')

  # Add recipe data to interaction data
  consolidated_data = []
  data_to_use = []
  recipe_indexes = {}

  if all_data:
    data_to_use = interaction_data
  else:
    data_to_use = interaction_data[:10000]

  for interaction in data_to_use:
    recipe_id = interaction[0]
    recipe_index = np.where(recipe_data[:,1] == recipe_id)[0]

    if len(recipe_index) != 0 and recipe_id not in recipe_indexes.keys():
      recipe_index = recipe_index[0]
      recipe_indexes[recipe_id] = recipe_index
    else:
      recipe_index = recipe_indexes[recipe_id]

    recipe = recipe_data[recipe_index]
    # Normalize rating to be between 0 and 1, with 1 representing the most difficult
    interaction[1] = 1 - (interaction[1] / 5)

    # Normalize minutes to be between 0 and 1, with 1 representing the most difficult
    if (recipe[2] != 0):
      recipe[2] = 1 - (1 / recipe[2])

    # Normalize n_steps to be between 0 and 1, with 1 representing the most difficult
    if (recipe[3] != 0):
      recipe[3] = 1 - (1 / recipe[3])

    # Normalize n_ingredients to be between 0 and 1, with 1 representing the most difficult
    if (recipe[4] != 0):
      recipe[4] = 1 - (1 / recipe[4])

    # Do not include id column from recipe data, since it is included in interaction data
    consolidated_data.append(np.concatenate([interaction, [recipe[0]], recipe[2:]]))

  return np.array(consolidated_data)

def averageRecipeData(recipe_data, verbose):
  recipe_ids = recipe_data[:,0]
  recipe_ids = np.unique(recipe_ids)
  average_recipe_data = []

  print('Averaging recipe data...')

  for recipe_id in recipe_ids:
    recipe = []
    recipe_rating = 0
    recipe_minutes = 0
    recipe_n_steps = 0
    recipe_n_ingredients = 0
    recipe_review_rating = 0
    recipe_review_no_stop_words_num_words = 0
    recipe_review_sentiment = 0
    recipe_indexes = np.where(recipe_data[:,0] == recipe_id)[0]

    for index in recipe_indexes:
      recipe = recipe_data[index]
      recipe_rating += recipe[1]
      recipe_minutes += recipe[2]
      recipe_n_steps += recipe[3]
      recipe_n_ingredients += recipe[4]
      recipe_review_rating += recipe[5]
      recipe_review_no_stop_words_num_words += recipe[6]
      recipe_review_sentiment += recipe[7]

    recipe_rating /= len(recipe_indexes)
    recipe_minutes /= len(recipe_indexes)
    recipe_n_steps /= len(recipe_indexes)
    recipe_n_ingredients /= len(recipe_indexes)
    recipe_review_rating /= len(recipe_indexes)
    recipe_review_no_stop_words_num_words /= len(recipe_indexes)
    recipe_review_sentiment /= len(recipe_indexes)
    average_recipe_data.append(np.array([recipe_id, recipe_rating, recipe_minutes, recipe_n_steps, recipe_n_ingredients, recipe_review_rating, recipe_review_no_stop_words_num_words, recipe_review_sentiment], dtype=np.float32))

  if verbose:
    print('Average recipe data shape: ' + str(np.array(average_recipe_data).shape))

  return np.array(average_recipe_data)
