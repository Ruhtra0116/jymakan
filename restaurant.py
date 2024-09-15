import streamlit as st
import pandas as pd
import gdown
import ast
import random
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the data from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Load emotion detection model and tokenizer
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-classification", model=model_name, top_k=None)
    return model, tokenizer

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model, tokenizer):
    max_length = 512  # Max token length for the model
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=max_length)
    
    try:
        emotions = emotion_model(lyrics[:tokenizer.model_max_length])
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        emotions = []
    return emotions

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

def extract_youtube_url(media_str):
    """Extract the YouTube URL from the Media field."""
    try:
        media_list = ast.literal_eval(media_str)  # Safely evaluate the string to a list
        for media in media_list:
            if media.get('provider') == 'youtube':
                return media.get('url')
    except (ValueError, SyntaxError):
        return None

# Recommend similar songs based on lyrics and detected emotions
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Load emotion detection model and tokenizer
    emotion_model, tokenizer = load_emotion_model()

    # Detect emotions in the selected song
    emotions = detect_emotions(song_lyrics, emotion_model, tokenizer)
    st.write(f"### Detected Emotions in {selected_song}:")
    
    if emotions and len(emotions) > 0:
        # Extract the emotions list from the first item
        emotion_list = emotions[0]
        
        # Find the emotion with the highest score
        if isinstance(emotion_list, list) and len(emotion_list) > 0:
            top_emotion = max(emotion_list, key=lambda x: x['score'])
            emotion_sentence = f"The emotion of the song is **{top_emotion['label']}**."
        else:
            emotion_sentence = "No emotions detected."
        
        st.write(emotion_sentence)
    else:
        st.write("No emotions detected.")

    # Compute lyrics similarity
    similarity_scores = compute_similarity(df, song_lyrics)

    # Add similarity scores to the dataframe
    df['similarity'] = similarity_scores

    # Exclude the selected song from recommendations
    df = df[df['Song Title'] != selected_song]

    # Recommend top N similar songs
    recommended_songs = df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL', 'Media']]

def display_random_songs(df, n=5):
    random_songs = df.sample(n=n)
    st.write("### Discover Songs:")
    for idx, row in random_songs.iterrows():
        youtube_url = extract_youtube_url(row.get('Media', ''))
        if youtube_url:
            # If a YouTube URL is available, make the song title a clickable hyperlink
            song_title = f"<a href='{youtube_url}' target='_blank' style='color: #FA8072; font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</a>"
        else:
            # If no YouTube URL, just display the song title
            song_title = f"<span style='font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</span>"

        with st.container():
            st.markdown(song_title, unsafe_allow_html=True)
            st.markdown(f"**Artist:** {row['Artist']}")
            st.markdown(f"**Album:** {row['Album']}")
            st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            st.markdown("---")

def main():
    # Add custom CSS to change the background image
    st.markdown(
        """
        <style>
        .main {
            background-image: url('data:image/jpeg;base64,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');
            background-size: cover;
            background-position: center;
        }

        .main::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.9); /* White overlay with 50% opacity */
        z-index: 0; /* Make sure the overlay sits behind the content */
        }
    
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: black;
            font-weight: 700;
            text-align: center;
        }
        .stButton>button {
            background-color: #fa8072;
            color: white;
            border-radius: 10px;
        }
        .stTextInput input {
            border: 1px solid #fa8072;
            padding: 0.5rem;
        }
        .stTextInput label {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🎵 Song Recommender Based on Lyrics & Emotions 🎶")
    df = download_data_from_drive()

    # Drop duplicate entries based on 'Song Title', 'Artist', 'Album', and 'Release Date'
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Search bar for song name or artist
    search_term = st.text_input("Search for a Song or Artist 🎤").strip()

    if search_term:
        # Filter by song title or artist name
        filtered_songs = df[
            (df['Song Title'].str.contains(search_term, case=False, na=False)) |
            (df['Artist'].str.contains(search_term, case=False, na=False))
        ]

        filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

        if filtered_songs.empty:
            st.write("No songs found matching the search term.")
        else:
            st.write(f"### Search Results for: {search_term}")
            for idx, row in filtered_songs.iterrows():
                with st.container():
                    st.markdown(f"<h2 style='font-weight: bold;'> {idx + 1}. {row['Song Title']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Artist:* {row['Artist']}")
                    st.markdown(f"*Album:* {row['Album']}")

                    if pd.notna(row['Release Date']):
                        st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"*Release Date:* Unknown")

                    song_url = row.get('Song URL', '')
                    if pd.notna(song_url) and song_url:
                        st.markdown(f"[View Lyrics on Genius]({song_url})")

                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    with st.expander("Show/Hide Lyrics"):
                        formatted_lyrics = row['Lyrics'].strip().replace('\n', '\n\n')
                        st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{formatted_lyrics}</pre>", unsafe_allow_html=True)
                    st.markdown("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations 🎧", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                
                for idx, row in enumerate(recommendations.iterrows(), 1):
                    st.markdown(f"<h2 style='font-weight: bold;'> {idx}. {row[1]['Song Title']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"*Artist:* {row[1]['Artist']}")
                    st.markdown(f"*Album:* {row[1]['Album']}")

                    if pd.notna(row[1]['Release Date']):
                        st.markdown(f"*Release Date:* {row[1]['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"*Release Date:* Unknown")

                    st.markdown(f"*Similarity Score:* {row[1]['similarity']:.2f}")

                    youtube_url = extract_youtube_url(row[1].get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    st.markdown("---")
    else:
        # Display random songs if no search term is provided
        display_random_songs(df)

if __name__ == '__main__':
    main()
