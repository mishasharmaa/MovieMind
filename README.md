# MovieMind – AI Movie Recommendation System

## Overview
MovieMind is a machine learning-powered movie recommendation web application that helps users discover movies in two ways:

Content-Based Recommendations – Find movies similar to a selected title
Genre-Based Browsing – Explore movies by category (like Netflix)

The system uses Natural Language Processing (NLP) and cosine similarity to generate intelligent recommendations based on movie metadata.

## Features
Dual Recommendation Modes
- Movie-Based Mode
    - Select a movie (e.g., Avatar)
    - Get similar movies based on content similarity
- Genre-Based Mode
    - No movie required
    - Browse movies by genre (e.g., Thriller, Action, Comedy)

## Machine Learning
- Uses TF-IDF Vectorization to convert text into meaningful numerical features
- Applies Cosine Similarity to measure similarity between movies
- Combines multiple features:
    - Plot overview
    - Genres
    - Keywords
    - Cast
    - Director

## User Interface
- Built with Streamlit
- Clean, modern dark UI (Netflix-inspired)
- Responsive movie grid layout
- Interactive navigation between:
    - Home page
    - Movie details page

## Performance Optimizations
```python @st.cache_data``` used for:
    - Dataset loading
    - Similarity computation
    - API calls
    - Reduces latency and improves responsiveness

## API Integration
- Uses TMDB (The Movie Database) API
- Fetches real-time:
    - Posters
    - Ratings
    - Runtime
    - Cast & crew

## Tech Stack
| Category        | Tools |
|----------------|------|
| Language       | Python |
| Data Processing| Pandas |
| NLP            | Scikit-learn (TF-IDF) |
| Frontend       | Streamlit |
| API            | TMDB API |
| Environment    | python-dotenv |