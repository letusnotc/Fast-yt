from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import dateparser  # Import dateparser
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Load RoBERTa model for sentiment analysis
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Function to analyze sentiment using RoBERTa
def analyze_sentiment(text):
    """Analyzes sentiment of a given text using RoBERTa."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = LABELS[probs.argmax().item()]
    return sentiment

# Function to convert YouTube relative time (e.g., "1 day ago") to actual datetime
def convert_to_datetime(relative_time):
    """Converts relative time like '2 days ago' to actual datetime."""
    parsed_date = dateparser.parse(relative_time)
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d %H:%M:%S")  # Format to readable date
    return "Unknown"

# Pydantic model for request
class VideoURL(BaseModel):
    url: str

# Endpoint to fetch and analyze YouTube comments
@app.post("/analyze/")
def analyze_youtube_comments(video: VideoURL):
    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video.url, sort_by=SORT_BY_RECENT)

        # Store data in a list
        data = []

        for comment in comments:
            text = comment['text']
            likes = comment.get('votes', 0)
            sentiment = analyze_sentiment(text)
            hearted = comment.get('heart', False)
            replies = comment.get('reply_count', 0) or (1 if comment.get('reply', False) else 0)
            date_time = convert_to_datetime(comment.get('time', "Unknown"))  # Convert date
            
            data.append({
                "comment": text,
                "sentiment": sentiment,
                "votes": likes,
                "hearted": hearted,
                "replies": replies,
                "date_time": date_time
            })
        
        # Convert to DataFrame and save as CSV (optional)
        df = pd.DataFrame(data)
        df.to_csv("youtube_comments_sentiment.csv", index=False)

        return {"message": "Sentiment analysis completed", "data": data[:100]}  # Return first 100 comments

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
