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

# ========================================================================
# PART C: ANALYZE AND VISUALIZE RESULTS
# ========================================================================
    
    print("\n" + "=" * 60)
    print("PART C: ANALYSIS AND VISUALIZATION")
    print("=" * 60)
    
    # Create output directory for plots
    import os
    os.makedirs('output', exist_ok=True)
    
    # 1. SENTIMENT DISTRIBUTION
    print("\n1. Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nPercentages:")
    print(sentiment_counts / len(df) * 100)
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    colors = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFC107'}
    sentiment_colors = [colors.get(s, '#999999') for s in sentiment_counts.index]
    
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=sentiment_colors, alpha=0.8)
    plt.title('Sentiment Distribution - Apple Vision Pro Reviews', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        plt.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ASPECT FREQUENCY ANALYSIS
    print("\n2. Aspect Frequency Analysis:")
    
    # Extract all aspects and their sentiments
    aspect_list = []
    aspect_sentiment_list = []
    
    for aspects_json in df['aspects']:
        if aspects_json:
            aspects = json.loads(aspects_json)
            for aspect_dict in aspects:
                if isinstance(aspect_dict, dict):
                    aspect_list.append(aspect_dict.get('aspect', '').lower())
                    aspect_sentiment_list.append({
                        'aspect': aspect_dict.get('aspect', '').lower(),
                        'sentiment': aspect_dict.get('sentiment', 'neutral').lower()
                    })
    
    # Count aspect frequencies
    aspect_counter = Counter(aspect_list)
    top_aspects = aspect_counter.most_common(15)
    
    print("\nTop 15 Most Mentioned Aspects:")
    for aspect, count in top_aspects:
        print(f"  {aspect}: {count} mentions")
    
    # Plot aspect frequency
    if top_aspects:
        aspects_names = [a[0] for a in top_aspects]
        aspects_counts = [a[1] for a in top_aspects]
        
        plt.figure(figsize=(12, 8))
        plt.barh(aspects_names, aspects_counts, color='#2196F3', alpha=0.8)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Aspect', fontsize=12)
        plt.title('Top 15 Most Mentioned Aspects', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        
        # Add count labels
        for i, count in enumerate(aspects_counts):
            plt.text(count, i, f' {count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('output/aspect_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. ASPECT SENTIMENT ANALYSIS
    print("\n3. Aspect Sentiment Analysis:")
    
    # Create DataFrame for aspect sentiments
    aspect_df = pd.DataFrame(aspect_sentiment_list)
    
    if not aspect_df.empty:
        # Get top aspects by frequency
        top_aspect_names = [a[0] for a in aspect_counter.most_common(10)]
        
        # Filter for top aspects
        aspect_df_filtered = aspect_df[aspect_df['aspect'].isin(top_aspect_names)]
        
        # Create crosstab
        aspect_sentiment_crosstab = pd.crosstab(
            aspect_df_filtered['aspect'], 
            aspect_df_filtered['sentiment']
        )
        
        print("\nSentiment breakdown by aspect:")
        print(aspect_sentiment_crosstab)
        
        # Plot stacked bar chart
        if not aspect_sentiment_crosstab.empty:
            plt.figure(figsize=(14, 8))
            aspect_sentiment_crosstab.plot(
                kind='barh', 
                stacked=True, 
                color={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FFC107'},
                figsize=(14, 8)
            )
            plt.xlabel('Number of Mentions', fontsize=12)
            plt.ylabel('Aspect', fontsize=12)
            plt.title('Sentiment Distribution by Aspect (Top 10)', fontsize=16, fontweight='bold')
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('output/aspect_sentiment.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. IDENTIFY POSITIVE ASPECTS
    print("\n4. Most Positively Mentioned Aspects:")
    positive_aspects = [a for a in aspect_sentiment_list if a['sentiment'] == 'positive']
    positive_counter = Counter([a['aspect'] for a in positive_aspects])
    top_positive = positive_counter.most_common(10)
    
    for aspect, count in top_positive:
        print(f"  ✓ {aspect}: {count} positive mentions")
    
    # 5. IDENTIFY NEGATIVE ASPECTS
    print("\n5. Most Negatively Mentioned Aspects:")
    negative_aspects = [a for a in aspect_sentiment_list if a['sentiment'] == 'negative']
    negative_counter = Counter([a['aspect'] for a in negative_aspects])
    top_negative = negative_counter.most_common(10)
    
    for aspect, count in top_negative:
        print(f"  ✗ {aspect}: {count} negative mentions")
    
    # Word Cloud for positive reviews
    positive_reviews = df[df['sentiment'] == 'Positive'][text_column]
    if not positive_reviews.empty:
        positive_text = ' '.join(positive_reviews.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            colormap='Greens').generate(positive_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Positive Reviews', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/wordcloud_positive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Word Cloud for negative reviews
    negative_reviews = df[df['sentiment'] == 'Negative'][text_column]
    if not negative_reviews.empty:
        negative_text = ' '.join(negative_reviews.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                            colormap='Reds').generate(negative_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Negative Reviews', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/wordcloud_negative.png', dpi=300, bbox_inches='tight')
        plt.close()