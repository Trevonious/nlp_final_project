import numpy as np

def analyzeReviewRatings(reviews, token_dictionary, recipe_data, reviews_no_stop_words, verbose):
  review_no_stop_words_num_words = []
  ratings = []

  print('Determining ratings of reviews...')

  print('Review length: ' + str(len(reviews)))

  for review in reviews:
    review_rating = 0

    for token in review:
      review_rating += token_dictionary[token]

    review_length = len(review)

    if review_length == 0:
      review_length = 20

    if review_rating == 0:
      review_rating = 1000

    # Normalize review length to be between 0 and 1, with 1 representing the most difficult
    normalized_review_length = 1 - (1 / review_length)
    review_no_stop_words_num_words.append(normalized_review_length)
    review_rating_average = review_rating / review_length
    # Normalize rating to be between 0 and 1, with 1 representing the most difficult
    final_review_rating = 1 / review_rating_average
    ratings.append(final_review_rating)

  ratings = np.array([ratings]).transpose()
  review_no_stop_words_num_words = np.array([review_no_stop_words_num_words]).transpose()
  data_with_review_ratings = np.concatenate([recipe_data, reviews_no_stop_words, ratings, review_no_stop_words_num_words], axis=1)

  return data_with_review_ratings
