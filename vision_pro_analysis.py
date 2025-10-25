import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from openai import OpenAI
import time
import json
from wordcloud import WordCloud
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# PART A: SENTIMENT ANALYSIS
# ============================================================================

def analyze_sentiment(review_text):
    """
    Analyze sentiment of a review using OpenAI's API.
    Returns: sentiment (Positive/Negative/Neutral)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of product reviews and respond with only one word: Positive, Negative, or Neutral."},
                {"role": "user", "content": f"Analyze the sentiment of this review: {review_text}"}
            ],
            temperature=0.3,
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip()
        return sentiment
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Neutral"
