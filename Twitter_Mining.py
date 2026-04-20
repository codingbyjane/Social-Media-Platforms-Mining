# Importing Dependencies

# Data Manipulation
import pandas as pd
import numpy as np

tweets_df = pd.read_json("data/raw/nikelululemonadidas_tweets.jsonl", lines=True)

# You can filter the dataframe to subsets of lululemon, nike, and adidas tweets using the 'brand' column after cleaningthe data

# Display basic info and properties of the dataframe
tweets_df.info()
print(f"\nDisplay the columns in the dataframe: {tweets_df.columns}")
print(f"\nFirst 5 rows of the 'full_text' column: {tweets_df['full_text'].head()}")
print(f"\nFirst row of the dataframe: {tweets_df.iloc[0]}")

# Extract attributes like id, created_at, retweet_count, text from the tweets and create a new dataframe with these attributes
tweets_subset_df = tweets_df[['id_str', 'created_at', 'retweet_count', 'full_text']]