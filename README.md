# Friday-Project-6
# Apple Vision Pro Customer Review Analysis

A comprehensive sentiment analysis and aspect extraction system for analyzing Apple Vision Pro customer reviews using OpenAI's API and Python.

## 📋 Project Overview

This project analyzes 80-100 customer reviews to:
- Perform sentiment analysis (Positive/Negative/Neutral)
- Extract specific product aspects mentioned in reviews
- Identify what customers like and dislike
- Generate visualizations and actionable insights
- Provide recommendations for product improvements

## 🎯 Features

- **Sentiment Classification**: Uses OpenAI's GPT-4o-mini to categorize each review
- **Aspect Extraction**: Identifies specific features mentioned (display, comfort, price, battery, etc.)
- **Sentiment by Aspect**: Analyzes whether aspects are mentioned positively or negatively
- **Visualizations**: 
  - Sentiment distribution bar chart
  - Most mentioned aspects
  - Aspect sentiment breakdown
  - Word clouds for positive and negative reviews
- **Insights Report**: Identifies strengths and areas for improvement

## 📦 Prerequisites

### Required Software
- Python 3.7 or higher
- pip (Python package manager)

### Required Python Libraries

Install all dependencies with:
```bash
pip install openai pandas matplotlib seaborn wordcloud python-dotenv
```

**Individual packages:**
- `openai` - For OpenAI API calls
- `pandas` - Data manipulation and analysis
- `matplotlib` - Creating visualizations
- `seaborn` - Enhanced statistical plots
- `wordcloud` - Generating word clouds
- `python-dotenv` - Loading environment variables

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
```bash
pip install openai pandas matplotlib seaborn wordcloud python-dotenv
```

### 3. Set Up OpenAI API Key

**Create a `.env` file** in the project root directory:
```
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

**Important Notes:**
- Get your API key from: https://platform.openai.com/api-keys
- **NO spaces** around the `=` sign
- **NO quotes** around the key
- Keep this file **private** - never commit it to GitHub

### 4. Prepare Your Database

Ensure you have a `feedback.db` SQLite database with:
- A table named `reviews`
- Two columns: `id` and `review_text`

## 📁 Project Structure

```
project-folder/
├── vision_pro_analysis.py    # Main analysis script
├── feedback.db                # SQLite database with reviews
├── .env                       # API key (not committed to git)
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── output/                    # Generated outputs (created automatically)
    ├── sentiment_distribution.png
    ├── aspect_frequency.png
    ├── aspect_sentiment.png
    ├── wordcloud_positive.png
    ├── wordcloud_negative.png
    └── analyzed_reviews.csv
```


### Expected Runtime
- **80-100 reviews**: Approximately 2-3 minutes
- Progress updates will display in the console

### Console Output
The script will print:
1. Data loading confirmation
2. Progress for each review being analyzed
3. Sentiment distribution statistics
4. Top mentioned aspects
5. Positive and negative aspect summaries
6. Key insights and recommendations

## 📊 Output Files

All results are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `sentiment_distribution.png` | Bar chart showing positive/negative/neutral review counts |
| `wordcloud_positive.png` | Visual representation of words in positive reviews |
| `wordcloud_negative.png` | Visual representation of words in negative reviews |
| `analyzed_reviews.csv` | Complete dataset with sentiment and extracted aspects |

## 📈 Analysis Components

### Part A: Sentiment Analysis
- Analyzes each review individually using OpenAI's API
- Classifies as Positive, Negative, or Neutral
- Stores results in the dataframe

### Part B: Aspect Extraction
- Identifies specific features mentioned (display, comfort, price, apps, battery, design, performance, weight, controls, spatial computing, immersion, etc.)
- Tags each aspect with sentiment
- Returns structured JSON data

### Part C: Visualization & Analysis
- Creates 5 professional visualizations
- Calculates sentiment distribution percentages
- Identifies most and least mentioned aspects
- Analyzes aspect-sentiment relationships

### Part D: Insights & Recommendations
- Summarizes key findings
- Highlights product strengths
- Identifies areas for improvement
- Provides actionable recommendations

## 🔒 Security Notes

### What to Keep Private
- `.env` file (contains API key)
- Never share your OpenAI API key publicly
- Add `.env` to `.gitignore`

### What's Safe to Share
- `vision_pro_analysis.py` (your code)
- `feedback.db` (contains fake review data for educational purposes)
- Output visualizations
- `README.md`

## 🐛 Troubleshooting

### "No module named 'openai'"
```bash
pip install openai
```

### "API key not found"
- Verify `.env` file exists in the same directory as the script
- Check that the variable is named exactly `OPENAI_API_KEY`
- Ensure no spaces around the `=` sign

### "Rate limit exceeded"
- The script includes 0.5-second delays between API calls
- If you still hit limits, increase the `time.sleep()` values

### "Database file not found"
- Ensure `feedback.db` is in the same directory as the script
- Verify the database has a `reviews` table with `id` and `review_text` columns

## 💰 Cost Estimate

Using GPT-4o-mini:
- Approximately **$0.01 - $0.02** per 100 reviews
- 80-100 reviews ≈ **$0.01 - $0.02 total**

## 📝 Requirements.txt

Create a `requirements.txt` file for easy installation:
```
openai>=1.0.0
pandas>=1.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
wordcloud>=1.9.0
python-dotenv>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

