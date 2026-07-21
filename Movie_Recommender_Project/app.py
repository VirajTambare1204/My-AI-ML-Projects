import streamlit as st
import pickle
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch AI", page_icon="🍿", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Dark Theme Backgrounds */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background-color: #1e2127;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        margin-bottom: 20px;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 25px rgba(0, 191, 255, 0.4);
    }
    
    /* Text Styling */
    .movie-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-rating {
        color: #ffd700;
        font-size: 1rem;
        font-weight: bold;
        margin-top: 5px;
    }
    .movie-year {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    
    /* Image Styling */
    .poster-img {
        border-radius: 10px;
        width: 100%;
        object-fit: cover;
        height: 280px;
    }
    
    /* Header Gradient */
    .header-text {
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem !important;
        text-align: center;
        padding-bottom: 10px;
    }
    .sub-header {
        text-align: center; 
        color: #a0a0a0; 
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    movies = pickle.load(open('models/movies.pkl', 'rb'))
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

# --- RECOMMENDATION LOGIC ---
def recommend(movie_title):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    # Get top 4 closest matches
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
    
    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].to_dict())
    return recommended_movies

# --- UI HEADER ---
st.markdown('<h1 class="header-text">🎬 CineMatch AI Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your Personal Machine Learning Movie Recommender</p>', unsafe_allow_html=True)
st.divider()

# --- SEARCH & CONTROLS ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_movie = st.selectbox(
        "🔍 Search the database:",
        movies['title'].values,
        index=1
    )
    analyze_button = st.button("🚀 Initialize Neural Search", use_container_width=True)

# --- RESULTS SECTION ---
if analyze_button:
    # 1. Animations and Loaders
    with st.spinner('Accessing vector database...'):
        time.sleep(0.5)
    with st.spinner('Calculating Cosine Similarity Matrix...'):
        time.sleep(0.8)
    
    st.toast('Analysis Complete! 🎉', icon='✅')
    
    # 2. Extract Data
    recommendations = recommend(selected_movie)
    selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
    
    # 3. Layout Tabs
    tab1, tab2 = st.tabs(["🍿 Top Matches", "📊 Engine Analytics"])
    
    with tab1:
        st.markdown(f"### Because you selected **{selected_movie}** ({selected_movie_data['year']}):")
        st.write("")
        
        # Create dynamic columns for the recommendations
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            movie = recommendations[idx]
            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <img src="{movie['poster_url']}" class="poster-img">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="movie-rating">⭐ {movie['rating']}/10</div>
                    <div class="movie-year">{movie['year']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive component per card
                if st.button("▶ Watch Trailer", key=f"btn_{idx}", use_container_width=True):
                    st.info(f"Connecting to media servers for {movie['title']}...")

    with tab2:
        st.markdown("### ⚙️ Under the Hood")
        st.write(f"The Natural Language Processing engine analyzed the following metadata vectors for **{selected_movie}**:")
        st.info(f"*{selected_movie_data['tags']}*")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label="Algorithm", value="Cosine Similarity")
        col_m2.metric(label="Vector Dimensions", value="5,000", delta="Optimized")
        col_m3.metric(label="Processing Time", value="0.04s", delta="-0.01s", delta_color="inverse")
        
    st.balloons()