# 🎬 IMDB Sentiment Analyzer

A modern, full-stack Natural Language Processing (NLP) web application that classifies movie reviews into **Positive** or **Negative** sentiments using Machine Learning, FastAPI, and an integrated SQLite database.

---

## 🌟 Features

- **Machine Learning Classification**: Accurately predicts sentiment using a pre-trained scikit-learn classifier with text vectorization.
- **NLP Preprocessing Pipeline**: Cleans and normalizes text with regex filtering, lowercase transformation, NLTK stopword elimination, and Porter stemming.
- **FastAPI Backend**: Asynchronous, high-performance REST API with automatic interactive documentation via Swagger UI (`/docs`).
- **Interactive Web Interface**: Clean, responsive frontend with real-time feedback and dynamic sentiment badge styling.
- **Database History Tracking**: Automatically persists submitted reviews, predictions, and timestamps to an SQLite database (`sentiment.db`).
- **Database Search & Sentiment Filtering**: Search positive and negative reviews directly using keyword queries and sentiment filters from both the web UI (`/data`) and REST API (`/api/reviews`).
- **XSS & Injection Protection**: HTML escaping and parameterized SQL queries protect stored records against vulnerabilities.
- **Cloud-Ready**: Includes `render.yaml` configuration for seamless one-click deployment on Render.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)
- **Machine Learning & NLP:** [Scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/), [Joblib](https://joblib.readthedocs.io/)
- **Database:** SQLite3
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
- **Deployment:** Render (`render.yaml`)

---

## 📂 Project Structure

```text
sentiment-analyzer/
├── .gitignore          # Ignored files (venv, cache, raw dataset, local DB)
├── render.yaml         # Render cloud deployment specification
├── requirements.txt    # Python package dependencies
├── app.py              # FastAPI application & SQLite persistence
├── index.html          # Frontend single-page application
├── model.pkl           # Trained sentiment classification model
├── vectorizer.pkl      # Pre-fitted text vectorizer
├── train.ipynb         # Model training & evaluation notebook
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/GabimaruT/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn app:app --reload
```
Alternatively:
```bash
python app.py
```

The application will be running at: **`http://localhost:8000`**

---

## 📡 API Endpoints

| Method | Endpoint | Description | Query Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web interface with analyzer and quick DB search | - |
| `POST` | `/predict` | Predict sentiment and save record into SQLite DB | - |
| `GET` | `/data` | Interactive Web UI to search & filter database records | `sentiment` (`positive`/`negative`), `q` (keyword) |
| `GET` | `/api/reviews` | JSON API endpoint to search stored reviews | `sentiment` (`positive`/`negative`), `q` (keyword), `limit` |
| `GET` | `/docs` | Interactive Swagger API documentation | - |
| `GET` | `/redoc` | Alternative ReDoc API documentation | - |

---

### 🔍 Search Examples

#### 1. Filter Positive Reviews (JSON API)
```bash
curl "http://localhost:8000/api/reviews?sentiment=positive"
```

#### 2. Filter Negative Reviews with Keyword Search (JSON API)
```bash
curl "http://localhost:8000/api/reviews?sentiment=negative&q=boring"
```

#### 3. Web UI Filtering
- **All Reviews:** `http://localhost:8000/data`
- **Positive Only:** `http://localhost:8000/data?sentiment=positive`
- **Negative Only:** `http://localhost:8000/data?sentiment=negative`
- **Keyword + Sentiment Search:** `http://localhost:8000/data?sentiment=positive&q=masterpiece`

---

## ☁️ Deployment on Render

This repository includes a `render.yaml` configuration. To deploy:

1. Push this repository to GitHub.
2. Log in to [Render](https://render.com/).
3. Create a **New Blueprints** project and connect your GitHub repository.
4. Render will automatically read `render.yaml`, install dependencies, and launch the service.

---

## 📝 License

Distributed under the MIT License. Feel free to use and adapt this project!
