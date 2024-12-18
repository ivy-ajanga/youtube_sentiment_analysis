import streamlit as st
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Download NLTK dependencies
nltk.download('vader_lexicon')

# Load the environment variables from the .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv('YOUTUBE_API_KEY')

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    query = urlparse(url).query
    params = parse_qs(query)
    return params.get('v', [None])[0]

# Function to extract YouTube comments
def get_youtube_comments(api_key, video_id, max_results=50):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        return comments
    except Exception as e:
        return str(e)

# Function for sentiment analysis
def analyze_sentiment(comments):
    sia = SentimentIntensityAnalyzer()
    results = []
    for comment in comments:
        score = sia.polarity_scores(comment)
        sentiment = 'Positive' if score['compound'] > 0 else 'Negative' if score['compound'] < 0 else 'Neutral'
        results.append({'Comment': comment, 'Sentiment': sentiment, 'Score': score['compound']})
    return pd.DataFrame(results)

# Streamlit App
st.title("YouTube Comment Sentiment Analysis")

# Input: YouTube URL
video_url = st.text_input("Paste the YouTube video URL here:", "")

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.write(f"Extracted Video ID: {video_id}")
        
        # Fetch Comments
        st.write("Fetching comments...")
        comments = get_youtube_comments(API_KEY, video_id)
        
        if isinstance(comments, str):  # Handle API errors
            st.error(f"Error: {comments}")
        elif comments:
            st.write(f"Fetched {len(comments)} comments.")
            
            # Perform Sentiment Analysis
            st.write("Performing sentiment analysis...")
            sentiment_df = analyze_sentiment(comments)
            
            # Display Results
            st.write("### Sentiment Analysis Results")
            st.dataframe(sentiment_df)
            
            # Generate Word Cloud
            st.write("### Word Cloud")
            text = " ".join(comment for comment in sentiment_df['Comment'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            # Plot Word Cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
            # Display Sentiment Distribution
            st.write("### Sentiment Distribution")
            sentiment_counts = sentiment_df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
        else:
            st.warning("No comments found for this video.")
    else:
        st.error("Invalid YouTube URL. Please check and try again.")
