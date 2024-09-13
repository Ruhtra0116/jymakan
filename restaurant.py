import streamlit as st
import pandas as pd
import gdown
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
    emotion_model = pipeline("text-classification", model=model_name, top_k=None)
    return emotion_model

# Detect emotions in the song lyrics
def detect_emotions(lyrics, emotion_model):
    try:
        # Use the pipeline to predict emotions from the lyrics
        emotions = emotion_model(lyrics[:512])  # Pass lyrics as a string
        if emotions:
            return [emotion['label'] for emotion in emotions]
        else:
            return []  # No emotions detected
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        return []

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Recommend similar songs based on lyrics and detected emotions
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Load emotion detection model
    emotion_model = load_emotion_model()

    # Detect emotions in the selected song
    selected_song_emotions = detect_emotions(song_lyrics, emotion_model)
    
    if not selected_song_emotions:
        st.write(f"No emotions detected in the selected song: {selected_song}.")
        return []

    # Detect emotions for all songs in the dataset
    df['detected_emotions'] = df['Lyrics'].apply(lambda lyrics: detect_emotions(lyrics, emotion_model))

    # Filter the songs with matching emotions
    matched_songs = df[df['detected_emotions'].apply(lambda emotions: set(emotions) == set(selected_song_emotions))]

    if matched_songs.empty:
        st.write(f"No recommendations found for {selected_song}.")
        return []

    # Compute similarity scores
    similarity_scores = compute_similarity(matched_songs, song_lyrics)
    matched_songs['similarity'] = similarity_scores

    # Return top N recommendations based on similarity
    recommended_songs = matched_songs.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL']]

# Main function for the Streamlit app
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Similarity")
    df = download_data_from_drive()

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    
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
                st.markdown(f"**{idx + 1}. {row['Song Title']} by {row['Artist']}**")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                st.write(f"### Recommended Songs Similar to {selected_song}")
                
                for idx, row in recommendations.iterrows():
                    st.markdown(f"**{idx + 1}. {row['Song Title']}**")
                    st.markdown(f"**Artist:** {row['Artist']}")
                    st.markdown(f"**Album:** {row['Album']}")
                    
                    # Check if 'Release Date' is a datetime object before formatting
                    if pd.notna(row['Release Date']):
                        st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d')}")
                    else:
                        st.markdown(f"**Release Date:** Unknown")
                    
                    st.markdown(f"**Similarity Score:** {row['similarity']:.2f}")
                    st.markdown("---")

    else:
        st.write("Please enter a song name or artist to search.")

if __name__ == '__main__':
    main()
