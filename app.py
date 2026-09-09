import os
import re
import html
import sqlite3
import joblib
import nltk
import uvicorn
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Download NLTK data quietly if not present
nltk.download('stopwords', quiet=True)

# ---------------- DATABASE CONFIGURATION (SQLite with Secured Queries) ----------------
DATABASE_FILE = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- FASTAPI SETUP & ML MODEL ----------------
app = FastAPI(title="IMDB Sentiment Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cv = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.lower()
    text = ''.join([i if i.isalnum() else ' ' for i in text])
    words = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

class ReviewRequest(BaseModel):
    review: str

# Serve Frontend Homepage
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# API endpoint: Predict and Save with Secured Parameterized Query ('?' placeholders)
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    processed = preprocess_text(data.review)
    vectorized = cv.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    sentiment_result = "Positive" if prediction == 1 else "Negative"
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Secured query execution: '?' placeholders prevent SQL injection
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reviews (review_text, sentiment, created_at) VALUES (?, ?, ?)",
        (data.review, sentiment_result, timestamp)
    )
    conn.commit()
    saved_id = cursor.lastrowid
    conn.close()

    return {
        "sentiment": sentiment_result,
        "code": int(prediction),
        "saved_id": saved_id
    }

# Web page endpoint: View Saved Data in a Table
@app.get("/data", response_class=HTMLResponse)
def view_saved_data():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Query execution with secure parameterized structure
    cursor.execute("SELECT id, review_text, sentiment, created_at FROM reviews ORDER BY id DESC")
    reviews = cursor.fetchall()
    conn.close()

    table_rows = ""
    for r in reviews:
        review_id = r[0]
        # HTML escape user content to prevent XSS vulnerabilities
        review_text = html.escape(str(r[1]))
        sentiment = html.escape(str(r[2]))
        created_at = html.escape(str(r[3]))
        table_rows += f"""
        <tr>
            <td>{review_id}</td>
            <td>{review_text}</td>
            <td><b>{sentiment}</b></td>
            <td>{created_at}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Saved Reviews Data</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            a {{ text-decoration: none; color: #007bff; font-weight: bold; margin-right: 10px; }}
        </style>
    </head>
    <body>
        <a href="/">&larr; Back to Analyzer</a> | 
        <a href="/docs">API Docs</a>
        <h2>Stored Sentiment Analysis Records</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Review Text</th>
                    <th>Sentiment Result</th>
                    <th>Timestamp (UTC)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)