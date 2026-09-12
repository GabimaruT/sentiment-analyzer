import os
import re
import html
import sqlite3
import joblib
import nltk
import uvicorn
from datetime import datetime
from typing import Optional
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
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
app = FastAPI(
    title="IMDB Sentiment Analyzer API",
    description="Analyze movie review sentiments and search historical records stored in SQLite database."
)

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

# Helper function to query reviews with secure parameterized filtering
def query_database_reviews(sentiment_filter: Optional[str] = None, search_query: Optional[str] = None, limit: Optional[int] = None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Query summary statistics
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM reviews WHERE LOWER(sentiment) = 'positive'")
    pos_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM reviews WHERE LOWER(sentiment) = 'negative'")
    neg_count = cursor.fetchone()[0]

    # Dynamic SQL with secure parameterized placeholders
    sql = "SELECT id, review_text, sentiment, created_at FROM reviews"
    conditions = []
    params = []

    if sentiment_filter and sentiment_filter.strip().lower() in ["positive", "negative"]:
        conditions.append("LOWER(sentiment) = ?")
        params.append(sentiment_filter.strip().lower())

    if search_query and search_query.strip():
        conditions.append("review_text LIKE ?")
        params.append(f"%{search_query.strip()}%")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY id DESC"

    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    reviews = [
        {
            "id": r[0],
            "review_text": r[1],
            "sentiment": r[2],
            "created_at": r[3]
        }
        for r in rows
    ]

    stats = {
        "total": total_count,
        "positive": pos_count,
        "negative": neg_count,
        "matched": len(reviews)
    }

    return reviews, stats

# API endpoint: Search reviews in DB by sentiment and keyword
@app.get("/api/reviews")
def api_search_reviews(
    sentiment: Optional[str] = Query(None, description="Filter by sentiment: 'positive', 'negative', or omit for all"),
    q: Optional[str] = Query(None, description="Search keyword in review text"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Max records to return")
):
    reviews, stats = query_database_reviews(sentiment_filter=sentiment, search_query=q, limit=limit)
    return {
        "stats": stats,
        "filters": {
            "sentiment": sentiment or "all",
            "q": q or ""
        },
        "reviews": reviews
    }

# Web page endpoint: View & Search Saved Data with Filters
@app.get("/data", response_class=HTMLResponse)
def view_saved_data(
    sentiment: Optional[str] = Query(None, description="Filter by sentiment ('positive' or 'negative')"),
    q: Optional[str] = Query(None, description="Search keyword")
):
    active_sentiment = (sentiment or "").strip().lower()
    search_text = (q or "").strip()
    
    reviews, stats = query_database_reviews(sentiment_filter=active_sentiment, search_query=search_text)

    # Build table rows with escaping
    if reviews:
        table_rows = ""
        for r in reviews:
            review_id = r["id"]
            review_text = html.escape(str(r["review_text"]))
            sentiment_val = html.escape(str(r["sentiment"]))
            created_at = html.escape(str(r["created_at"]))
            badge_class = "badge-positive" if sentiment_val.lower() == "positive" else "badge-negative"
            
            table_rows += f"""
            <tr>
                <td style="color: #666; font-weight: 500;">#{review_id}</td>
                <td style="line-height: 1.5;">{review_text}</td>
                <td><span class="badge {badge_class}">{sentiment_val}</span></td>
                <td style="color: #777; font-size: 0.9em; white-space: nowrap;">{created_at}</td>
            </tr>
            """
    else:
        table_rows = """
        <tr>
            <td colspan="4" style="text-align: center; padding: 30px; color: #888;">
                🔍 No reviews found matching your search criteria.
            </td>
        </tr>
        """

    # Escape filter text for input fields
    escaped_q = html.escape(search_text)
    
    # Selected state helpers
    all_selected = "selected" if active_sentiment not in ["positive", "negative"] else ""
    pos_selected = "selected" if active_sentiment == "positive" else ""
    neg_selected = "selected" if active_sentiment == "negative" else ""
    
    all_pill_active = "active" if active_sentiment not in ["positive", "negative"] else ""
    pos_pill_active = "active" if active_sentiment == "positive" else ""
    neg_pill_active = "active" if active_sentiment == "negative" else ""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Saved Reviews Database Search</title>
        <style>
            :root {{
                --primary: #007bff;
                --primary-hover: #0056b3;
                --bg: #f8f9fa;
                --card-bg: #ffffff;
                --text: #212529;
                --pos-bg: #d4edda;
                --pos-text: #155724;
                --neg-bg: #f8d7da;
                --neg-text: #721c24;
            }}
            * {{ box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; 
                margin: 0; 
                padding: 30px 20px; 
                background-color: var(--bg);
                color: var(--text);
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: var(--card-bg);
                padding: 25px 30px;
                border-radius: 10px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            }}
            .header-nav {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid #e9ecef;
            }}
            .nav-link {{
                text-decoration: none;
                color: var(--primary);
                font-weight: 600;
                font-size: 14px;
            }}
            .nav-link:hover {{ text-decoration: underline; }}
            h2 {{ margin: 0 0 8px 0; color: #1a1a1a; }}
            p.subtitle {{ margin: 0 0 20px 0; color: #6c757d; font-size: 14px; }}
            
            /* Stats Bar */
            .stats-bar {{
                display: flex;
                gap: 12px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .stat-pill {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                text-decoration: none;
                border: 1px solid #dee2e6;
                background: #fdfdfd;
                color: #495057;
                transition: all 0.2s ease;
            }}
            .stat-pill:hover, .stat-pill.active {{
                border-color: var(--primary);
                background: #e7f1ff;
                color: var(--primary-hover);
            }}
            .stat-pill.pill-pos.active {{
                background: var(--pos-bg);
                border-color: #28a745;
                color: var(--pos-text);
            }}
            .stat-pill.pill-neg.active {{
                background: var(--neg-bg);
                border-color: #dc3545;
                color: var(--neg-text);
            }}

            /* Search Form */
            .filter-form {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f1f3f5;
                border-radius: 8px;
            }}
            .filter-form input[type="text"] {{
                flex: 1;
                min-width: 220px;
                padding: 9px 12px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                font-size: 14px;
            }}
            .filter-form select {{
                padding: 9px 12px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                font-size: 14px;
                background-color: white;
            }}
            .filter-form button {{
                padding: 9px 18px;
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.2s;
            }}
            .filter-form button:hover {{
                background-color: var(--primary-hover);
            }}
            .filter-form a.btn-clear {{
                display: inline-flex;
                align-items: center;
                padding: 9px 14px;
                color: #6c757d;
                background: #e9ecef;
                border-radius: 5px;
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
            }}
            .filter-form a.btn-clear:hover {{
                background: #dee2e6;
                color: #333;
            }}

            /* Table Styles */
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin-top: 10px;
            }}
            th, td {{ 
                border-bottom: 1px solid #e9ecef; 
                padding: 12px 14px; 
                text-align: left; 
                font-size: 14px;
            }}
            th {{ 
                background-color: #f8f9fa; 
                font-weight: 600;
                color: #495057;
            }}
            tr:hover {{ background-color: #fcfcfc; }}
            
            /* Badges */
            .badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 0.3px;
            }}
            .badge-positive {{
                color: var(--pos-text);
                background-color: var(--pos-bg);
                border: 1px solid #c3e6cb;
            }}
            .badge-negative {{
                color: var(--neg-text);
                background-color: var(--neg-bg);
                border: 1px solid #f5c6cb;
            }}
            .result-summary {{
                font-size: 13px;
                color: #6c757d;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-nav">
                <a class="nav-link" href="/">&larr; Back to Sentiment Analyzer</a>
                <a class="nav-link" href="/docs" target="_blank">📖 Interactive API Docs</a>
            </div>

            <h2>Stored Sentiment Database Records</h2>
            <p class="subtitle">Search, inspect, and filter historical reviews and predicted sentiments.</p>

            <!-- Quick Filter Stats Pills -->
            <div class="stats-bar">
                <a href="/data" class="stat-pill {all_pill_active}">
                    📋 All Reviews <span>({stats["total"]})</span>
                </a>
                <a href="/data?sentiment=positive" class="stat-pill pill-pos {pos_pill_active}">
                    👍 Positive <span>({stats["positive"]})</span>
                </a>
                <a href="/data?sentiment=negative" class="stat-pill pill-neg {neg_pill_active}">
                    👎 Negative <span>({stats["negative"]})</span>
                </a>
            </div>

            <!-- Search and Filter Form -->
            <form class="filter-form" method="GET" action="/data">
                <input 
                    type="text" 
                    name="q" 
                    placeholder="Search keywords in reviews..." 
                    value="{escaped_q}"
                >
                <select name="sentiment">
                    <option value="" {all_selected}>All Sentiments</option>
                    <option value="positive" {pos_selected}>Positive Only</option>
                    <option value="negative" {neg_selected}>Negative Only</option>
                </select>
                <button type="submit">🔍 Search / Filter</button>
                <a href="/data" class="btn-clear">Reset</a>
            </form>

            <div class="result-summary">
                Showing <b>{stats["matched"]}</b> of <b>{stats["total"]}</b> total recorded reviews
            </div>

            <table>
                <thead>
                    <tr>
                        <th style="width: 80px;">ID</th>
                        <th>Review Text</th>
                        <th style="width: 120px;">Sentiment</th>
                        <th style="width: 170px;">Timestamp (UTC)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)