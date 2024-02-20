import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob

# Download necessary NLTK assets if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Load the review data
data = pd.read_csv('reviews.csv')


# Preprocessing functions
def preprocess_text(text):
  # Tokenization
  tokens = nltk.word_tokenize(text)

  # Stop word removal
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word.lower() not in stop_words]

  # Stemming
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(word) for word in tokens]

  return ' '.join(tokens)


# Apply preprocessing
data['processed_text'] = data['review_text'].apply(preprocess_text)


# Sentiment analysis
def get_sentiment(text):
  analysis = TextBlob(text)
  return analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to +1 (positive)


data['sentiment'] = data['processed_text'].apply(get_sentiment)

# Results
print(data.head())
