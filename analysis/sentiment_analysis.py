import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyzeReviewSentiment(data, verbose):
  reviews = data[:,7]
  sentiment = SentimentIntensityAnalyzer()
  sentiment_scores = []

  print('Determining sentiment of reviews...')

  for review in reviews:
    sentiment_score = sentiment.polarity_scores(review)
    # Flip sentiment score, then add 1 and divide by 2
    # This essentially makes values from 0 to 0.5 positive sentiment and 0.5 to 1.0 negative sentiment
    sentiment_scores.append(((sentiment_score.get('compound') * -1) + 1) / 2)

  sentiment_scores = np.array([sentiment_scores]).transpose()
  data_with_sentiment_scores = np.append(data, sentiment_scores, axis=1)
  
  return data_with_sentiment_scores
