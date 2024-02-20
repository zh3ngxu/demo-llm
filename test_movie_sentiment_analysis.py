import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews

# Download necessary resources for NLTK
nltk.download('vader_lexicon')
nltk.download('movie_reviews')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load movie reviews from NLTK corpus
reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# Analyze sentiment for each review
positive_count = 0
negative_count = 0
total_count = len(reviews)

for review, category in reviews:
  sentiment_score = sia.polarity_scores(review)['compound']
  if sentiment_score >= 0.05:
    positive_count += 1
  elif sentiment_score <= -0.05:
    negative_count += 1

# Calculate sentiment analysis results
positive_percentage = (positive_count / total_count) * 100
negative_percentage = (negative_count / total_count) * 100

# Print results
print("Sentiment Analysis Results:")
print("Total Reviews:", total_count)
print("Positive Reviews: {:.2f}%".format(positive_percentage))
print("Negative Reviews: {:.2f}%".format(negative_percentage))
