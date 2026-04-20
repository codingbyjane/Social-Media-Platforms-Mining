# Importing Dependencies

# Data Manipulation
import pandas as pd
import numpy as np
from collections import Counter

# Text Preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tweets_df = pd.read_json("data/raw/nikelululemonadidas_tweets.jsonl", lines=True)

# Display basic info and properties of the dataframe
tweets_df.info()
print(f"\nDisplay the columns in the dataframe: {tweets_df.columns}")
print(f"\nFirst 5 rows of the 'full_text' column: {tweets_df['full_text'].head()}")
print(f"\nFirst row of the dataframe: {tweets_df.iloc[0]}")

# Extract attributes like id, created_at, retweet_count, text from the tweets and create a new dataframe with these attributes
tweets_df = tweets_df[['id_str', 'created_at', 'retweet_count', 'full_text']].copy()

# Remove duplicates & Empty Rows
tweets_df.drop_duplicates(inplace=True)
tweets_df = tweets_df.dropna(inplace=True)

# Load the English stopword list
stopword_set = set(stopwords.words('english') + ['rt', 'amp', 'via'])

# Discard negation terms from the stopword list to preserve their meaning in sentiment analysis
stopword_set.discard('not')
stopword_set.discard('no')

# Text Precprocessing
def preprocess(text, tokenizer=TweetTokenizer(), lemmatizer=WordNetLemmatizer(), stopword_set=stopword_set):
    text = text.lower() # Casefolding
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters, punctuation, and numbers
    tokens = tokenizer.tokenize(text) # Tokenization
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopword_set] # Remove stopwords and lemmatize
    
    return tokens


# Entity Analysis




# Define a function to detect and extract brand mentions in the tweet text
def get_brand(tweet):
    casefolded_text = str(tweet['full_text']).lower()