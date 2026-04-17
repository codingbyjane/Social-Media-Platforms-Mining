# Importing Dependencies
import os
from dotenv import load_dotenv
import tweepy

# Loading Environment Variables
load_dotenv() # loads .env file

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

# Initializing Twitter API Client
client = tweepy.Client(bearer_token=bearer_token)