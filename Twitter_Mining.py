# Importing Dependencies

# Data Manipulation
import pandas as pd
import numpy as np

tweets_df = pd.read_json("data/raw/nikelululemonadidas_tweets.jsonl", lines=True)

# You can filter the dataframe to subsets of lululemon, nike, and adidas tweets using the 'brand' column after cleaningthe data

# Display basic info and properties of the dataframe
print(f"Basic Info: {tweets_df.info()}")

print(f"First 5 rows of the dataframe: {tweets_df.head()}")