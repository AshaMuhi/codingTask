import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

# Load the dataset
file_path = 'Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv'
data = pd.read_csv(file_path)

# Select the 'reviews.text' column and drop missing values
reviews_data = data['reviews.text']
clean_data = reviews_data.dropna().reset_index(drop=True)

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocess the text data
def preprocess_text(text):
    # Convert to lower case and strip whitespace
    doc = nlp(text.lower().strip())
    # Lemmatize and remove stop words
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
clean_data = clean_data.apply(preprocess_text)

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Test the sentiment analysis function on a few sample product reviews
sample_reviews = clean_data.sample(5).tolist()
sample_sentiments = [analyze_sentiment(review) for review in sample_reviews]

for review, sentiment in zip(sample_reviews, sample_sentiments):
    print(f"Review: {review}\nSentiment: {sentiment}\n")

# Choose two product reviews from the 'reviews.text' column
review1 = clean_data[0]
review2 = clean_data[1]

# Compare their similarity
doc1 = nlp(review1)
doc2 = nlp(review2)
similarity = doc1.similarity(doc2)

print(f"Review 1: {review1}")
print(f"Review 2: {review2}")
print(f"Similarity: {similarity}")
