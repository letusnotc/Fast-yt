from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import pandas as pd
import torch
import dateparser
from datetime import datetime
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load RoBERTa sentiment analysis model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Function to analyze sentiment using RoBERTa
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return LABELS[probs.argmax().item()]

# Function to convert relative time to actual datetime
def convert_to_datetime(relative_time):
    parsed_date = dateparser.parse(relative_time)
    return parsed_date.strftime("%Y-%m-%d %H:%M:%S") if parsed_date else "Unknown"

# Pydantic model for request
class VideoURL(BaseModel):
    url: str

# Test endpoint for debugging
@app.get("/")
def home():
    return {"message": "FastAPI YouTube Sentiment Analysis is Running!"}

# Endpoint to fetch and analyze YouTube comments
@app.post("/analyze/")
async def analyze_youtube_comments(video: VideoURL):
    try:
        logger.info(f"Processing URL: {video.url}")

        # Validate YouTube URL format
        if not video.url.startswith("http"):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Initialize downloader and fetch comments
        downloader = YoutubeCommentDownloader()
        comments = list(downloader.get_comments_from_url(video.url, sort_by=SORT_BY_RECENT))
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found.")

        # Process comments and analyze sentiment
        data = []
        for comment in comments:
            text = comment.get("text", "")
            sentiment = analyze_sentiment(text)
            date_time = convert_to_datetime(comment.get("time", "Unknown"))
            data.append({
                "comment": text,
                "sentiment": sentiment,
                "votes": comment.get("votes", 0),
                "hearted": comment.get("heart", False),
                "replies": comment.get("reply_count", 0),
                "date_time": date_time
            })

        # Convert processed data to a DataFrame and then save as CSV
        df = pd.DataFrame(data)
        csv_filename = "sentiment_analysis.csv"
        # This will overwrite any previous CSV file with the same name.
        df.to_csv(csv_filename, index=False, quoting=1)

        logger.info("Sentiment analysis completed and CSV file saved successfully!")

        # Return the CSV file as a downloadable response
        return FileResponse(
            csv_filename,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiment_analysis.csv"}
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
