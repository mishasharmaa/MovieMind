import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ================= CONFIG =================
st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="wide"
)

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

# ================= STYLE =================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
.block-container {
    padding-top: 0rem !important;
}
header {
    display: none !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}
div[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
}
section.main > div {
    padding-top: 0rem !important;
}
h1, h2, h3, p, label {
    color: white !important;
}
.card {
    background-color: #1c1f26;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    transition: 0.2s;
}
.card:hover {
    transform: scale(1.05);
}
.header-box {
    background-color: #1c1f26;
    padding: 0px 20px 15px 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header-box h1 {
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "results" not in st.session_state:
    st.session_state.results = None

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

    def fetch_director(text):
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

    return movies

movies = load_data()

# GENRE LIST
all_genres = sorted(list(set(g for sublist in movies['genres'] for g in sublist)))

# ================= SIMILARITY =================
@st.cache_data
def compute_similarity(_movies):
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    vectors = tfidf.fit_transform(_movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = compute_similarity(movies)

# ================= API =================
@st.cache_data(ttl=3600)
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&append_to_response=credits"
    data = requests.get(url).json()

    poster = "https://via.placeholder.com/500x750?text=No+Image"
    if data.get('poster_path'):
        poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path']

    rating = data.get("vote_average", "N/A")
    year = data.get("release_date", "N/A")[:4]
    runtime = data.get("runtime", "N/A")
    overview = data.get("overview", "N/A")

    director = "N/A"
    cast = []

    if "credits" in data:
        for crew in data["credits"]["crew"]:
            if crew["job"] == "Director":
                director = crew["name"]

        cast = [actor["name"] for actor in data["credits"]["cast"][:5]]

    return poster, rating, year, runtime, overview, director, cast

# ================= RECOMMEND =================
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:13]

    results = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        poster, rating, year, runtime, overview, director, cast = fetch_movie_details(movie_id)
        results.append((title, poster, rating, year, runtime, overview, director, cast, movie_id))

    return results

# ================= UI =================
if st.session_state.page == "home":

    st.markdown("<div class='header-box'>", unsafe_allow_html=True)

    st.title("🎬 MovieMind")
    st.caption("AI-Powered Movie Recommendation System")

    st.markdown("""
### 🎯 Discover Movies Smarter
Browse by movie OR explore genres like Netflix.
""")

    st.markdown("</div>", unsafe_allow_html=True)

    # INPUT
    selected_movie = st.selectbox(
        "Choose a movie (optional)",
        ["None"] + list(movies['title'].values)
    )

    sort_option = st.selectbox("Mode", ["Movie Recommendation", "Browse by Genre"])

    selected_genre = None
    if sort_option == "Browse by Genre":
        selected_genre = st.selectbox("Select Genre", all_genres)

    if st.button("Get Results"):

        # 🎬 Movie mode
        if sort_option == "Movie Recommendation" and selected_movie != "None":
            st.session_state.results = recommend(selected_movie)

        # 🎯 Genre mode
        elif sort_option == "Browse by Genre" and selected_genre:
            genre_results = []

            for i in range(len(movies)):
                genres = movies.iloc[i]['genres']

                if selected_genre in genres:
                    movie_id = movies.iloc[i].movie_id
                    title = movies.iloc[i].title

                    poster, rating, year, runtime, overview, director, cast = fetch_movie_details(movie_id)

                    genre_results.append((title, poster, rating, year, runtime, overview, director, cast, movie_id))

                if len(genre_results) == 20:
                    break

            st.session_state.results = genre_results

        else:
            st.warning("Please select a movie or choose a genre")

    # RESULTS
    if st.session_state.results:

        results = st.session_state.results

        num_cols = 5
        rows = [results[i:i+num_cols] for i in range(0, len(results), num_cols)]

        for row_idx, row in enumerate(rows):
            cols = st.columns(num_cols)

            for col_idx, item in enumerate(row):
                title, poster, rating, year, runtime, overview, director, cast, movie_id = item

                with cols[col_idx]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.image(poster)

                    if st.button(title, key=f"{row_idx}_{col_idx}"):
                        st.session_state.movie_id = movie_id
                        st.session_state.page = "details"
                        st.rerun()

                    st.markdown(f"⭐ **{rating}**  \n📅 {year}")
                    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "details":

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    poster, rating, year, runtime, overview, director, cast = fetch_movie_details(st.session_state.movie_id)

    title = movies[movies['movie_id']==st.session_state.movie_id].iloc[0].title

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(poster)

    with col2:
        st.title(title)
        st.write(f"⭐ Rating: {rating}")
        st.write(f"📅 Release: {year}")
        st.write(f"⏱ Runtime: {runtime} min")
        st.write(f"🎬 Director: {director}")

        if cast:
            st.write("🎭 Cast: " + ", ".join(cast))

        st.markdown("### Overview")
        st.write(overview)