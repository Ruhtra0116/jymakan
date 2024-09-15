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
        background-color: rgba(255, 255, 255, 0.5); /* White overlay with 50% opacity */
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
