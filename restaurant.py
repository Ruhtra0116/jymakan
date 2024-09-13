import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

# Load the emotion model and tokenizer
@st.cache_resource
def load_emotion_model():
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return model, tokenizer

# Detect emotions for song lyrics
def detect_emotions(lyrics, model, tokenizer, max_length=512):
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=max_length)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=1)
    return probabilities.detach().numpy()

# Precompute emotions for all songs in the dataset
@st.cache_data
def precompute_emotions(df):
    emotion_model, tokenizer = load_emotion_model()

    def compute_emotion_for_song(lyrics):
        if not isinstance(lyrics, str) or len(lyrics) == 0:
            return np.zeros(7)  # Assuming the emotion model outputs a 7-dim vector
        return detect_emotions(lyrics, emotion_model, tokenizer)[0]

    df['detected_emotions'] = df['Lyrics'].apply(compute_emotion_for_song)
    return df

# Check if two songs share similar emotions
def has_matching_emotion(emotions1, emotions2, threshold=0.5):
    similarity = cosine_similarity([emotions1], [emotions2])[0][0]
    return similarity >= threshold

# Compute similarity between song lyrics (can be enhanced using advanced NLP techniques)
def compute_similarity(df, selected_lyrics):
    selected_lyrics_vector = np.array([selected_lyrics])
    song_vectors = df['Lyrics'].apply(lambda lyrics: np.array([lyrics]))
    similarities = [cosine_similarity(selected_lyrics_vector, song_vector)[0][0] for song_vector in song_vectors]
    return similarities

# Recommend similar songs based on lyrics and detected emotions
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]
    selected_song_emotions = song_data['detected_emotions'].values[0]

    # Filter songs based on matching detected emotions
    filtered_df = df[df['detected_emotions'].apply(lambda emotions: has_matching_emotion(emotions, selected_song_emotions))]

    # If no songs match the emotion, notify the user
    if filtered_df.empty:
        st.write("No songs found with similar emotions.")
        return []

    # Compute lyrics similarity
    similarity_scores = compute_similarity(filtered_df, song_lyrics)

    # Recommend top N similar songs with matching emotions
    filtered_df['similarity'] = similarity_scores
    recommended_songs = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL', 'Media']]

# Extract YouTube URL from media link
def extract_youtube_url(media_link):
    if isinstance(media_link, str) and 'youtube' in media_link:
        return media_link
    return None

# Main Streamlit app
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Similarity")
    
    # Load dataset from a hypothetical external source
    df = download_data_from_drive()

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Precompute emotions for all songs in the dataset (cached for faster access)
    df = precompute_emotions(df)

    # Search bar for song name or artist
    search_term = st.text_input("Enter a Song Name or Artist").strip()

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
                    st.markdown(f"**Artist:** {row['Artist']}")
                    st.markdown(f"**Album:** {row['Album']}")

                    if pd.notna(row['Release Date']):
                        st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"**Release Date:** Unknown")
                    
                    # Display link to Genius.com page if URL is available
                    song_url = row.get('Song URL', '')
                    if pd.notna(song_url) and song_url:
                        st.markdown(f"[View Lyrics on Genius]({song_url})")

                    # Extract and display YouTube video if URL is available
                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    with st.expander("Show/Hide Lyrics"):
                        formatted_lyrics = row['Lyrics'].strip().replace('\n', '\n\n')
                        st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{formatted_lyrics}</pre>", unsafe_allow_html=True)
                    st.markdown("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                for idx, row in recommendations.iterrows():
                    st.markdown(f"**No. {idx + 1}: {row['Song Title']}**")
                    st.markdown(f"**Artist:** {row['Artist']}")
                    st.markdown(f"**Album:** {row['Album']}")

                    if pd.notna(row['Release Date']):
                        st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"**Release Date:** Unknown")

                    st.markdown(f"**Similarity Score:** {row['similarity']:.2f}")

                    # Extract and display YouTube video if URL is available
                    youtube_url = extract_youtube_url(row.get('Media', ''))
                    if youtube_url:
                        video_id = youtube_url.split('watch?v=')[-1]
                        st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                    st.markdown("---")

    else:
        st.write("Please enter a song name or artist to search.")

# Function to download data from external source (placeholder)
@st.cache_data
def download_data_from_drive():
    # This is a placeholder function to simulate loading data
    # Replace with actual logic to download your dataset
    data = {
        "Song Title": ["Song A", "Song B", "Song C"],
        "Artist": ["Artist 1", "Artist 2", "Artist 3"],
        "Album": ["Album 1", "Album 2", "Album 3"],
        "Release Date": ["2021-01-01", "2020-05-05", "2019-11-11"],
        "Lyrics": ["Lyrics of Song A", "Lyrics of Song B", "Lyrics of Song C"],
        "Song URL": ["https://genius.com/SongA", "https://genius.com/SongB", "https://genius.com/SongC"],
        "Media": ["https://youtube.com/watch?v=videoA", "https://youtube.com/watch?v=videoB", "https://youtube.com/watch?v=videoC"]
    }
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    main()
