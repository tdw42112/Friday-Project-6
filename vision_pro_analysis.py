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

# ============================================================================
# PART B: ASPECT EXTRACTION
# ============================================================================

def extract_aspects(review_text):
    """
    Extract specific aspects/features mentioned in the review.
    Returns: list of aspects with their associated sentiment
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an expert at extracting product aspects from reviews. 
                Extract specific features/aspects mentioned in Apple Vision Pro reviews.
                Return a JSON list of objects with 'aspect' and 'sentiment' (positive/negative/neutral).
                Focus on: display, comfort, price, apps, battery, design, performance, weight, controls, spatial computing, immersion, etc."""},
                {"role": "user", "content": f"Extract aspects from this review: {review_text}"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        aspects_text = response.choices[0].message.content.strip()
        # Parse JSON response
        try:
            aspects = json.loads(aspects_text)
            return aspects
        except:
            # Fallback if JSON parsing fails
            return []
    except Exception as e:
        print(f"Error in aspect extraction: {e}")
        return []

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    # Load data from SQLite database
    print("Loading data from feedback.db...")
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Identify the review text column (adjust if needed)
    text_column = 'review_text' if 'review_text' in df.columns else df.columns[1]
    
    # Initialize columns for results
    df['sentiment'] = ''
    df['aspects'] = ''
    
    # PART A: Perform sentiment analysis
    print("=" * 60)
    print("PART A: SENTIMENT ANALYSIS")
    print("=" * 60)
    
    for idx, row in df.iterrows():
        review_text = str(row[text_column])
        print(f"Analyzing review {idx + 1}/{len(df)}...")
        
        # Analyze sentiment
        sentiment = analyze_sentiment(review_text)
        df.at[idx, 'sentiment'] = sentiment
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    print("\nSentiment analysis complete!")
    
    # PART B: Extract aspects
    print("\n" + "=" * 60)
    print("PART B: ASPECT EXTRACTION")
    print("=" * 60)
    
    all_aspects = []
    
    for idx, row in df.iterrows():
        review_text = str(row[text_column])
        print(f"Extracting aspects from review {idx + 1}/{len(df)}...")
        
        # Extract aspects
        aspects = extract_aspects(review_text)
        df.at[idx, 'aspects'] = json.dumps(aspects)
        all_aspects.extend(aspects)
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    print("\nAspect extraction complete!")