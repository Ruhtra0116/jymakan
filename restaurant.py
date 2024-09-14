import streamlit as st
import pandas as pd
import gdown
import ast
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
    if not isinstance(lyrics, str) or lyrics.strip() == "":
        return []  # Return an empty list for invalid or empty lyrics
    
    try:
        max_length = 512  # Max token length for the model
        emotions = emotion_model(lyrics[:max_length])
        return emotions
    except Exception as e:
        st.write(f"Error in emotion detection for lyrics: {lyrics[:100]}...: {e}")
        return []

# Detect emotions for all songs in the dataset
def detect_emotions_for_songs(df):
    emotion_model, tokenizer = load_emotion_model()
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    
    # Apply emotion detection to each song's lyrics and store results in 'Emotions' column
    df['Emotions'] = df['Lyrics'].apply(lambda lyrics: detect_emotions(lyrics, emotion_model, tokenizer))
    return df

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Extract YouTube URL from media field
def extract_youtube_url(media_str):
    try:
        media_list = ast.literal_eval(media_str)
        for media in media_list:
            if media.get('provider') == 'youtube':
                return media.get('url')
    except (ValueError, SyntaxError):
        return None

# Recommend songs with the same detected emotion
def recommend_same_emotion_songs(df, top_emotion, selected_song, top_n=5):
    if 'Emotions' not in df.columns:
        st.write("No emotion data available for recommendations.")
        return pd.DataFrame()  # Return empty DataFrame if 'Emotions' column is missing
    
    # Filter songs with the same detected emotion
    emotion_recommendations = df[df['Emotions'].apply(lambda x: top_emotion['label'] in [e['label'] for e in x] if x else False)]
    emotion_recommendations = emotion_recommendations[emotion_recommendations['Song Title'] != selected_song]
    return emotion_recommendations.head(top_n)

# Recommend songs based on lyric similarity
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return pd.DataFrame()  # Return an empty DataFrame if no song is found
    
    song_lyrics = song_data['Lyrics'].values[0]
    
    # Detect emotions in the selected song
    emotion_model, tokenizer = load_emotion_model()
    emotions = detect_emotions(song_lyrics, emotion_model, tokenizer)
    if emotions and len(emotions) > 0:
        top_emotion = max(emotions[0], key=lambda x: x['score'])
        st.write(f"### Detected Emotion in {selected_song}: **{top_emotion['label']}**")
    else:
        st.write(f"### No emotions detected for {selected_song}.")
        return pd.DataFrame()

    # Recommend songs with the same emotion
    st.write(f"### Songs with the same emotion: **{top_emotion['label']}**")
    same_emotion_recommendations = recommend_same_emotion_songs(df, top_emotion, selected_song)
    if same_emotion_recommendations.empty:
        st.write("No songs found with the same emotion.")
    else:
        for idx, row in same_emotion_recommendations.iterrows():
            st.write(f"**{row['Song Title']}** by {row['Artist']} (Album: {row['Album']})")
    
    # Compute lyrics similarity for further recommendations
    st.write(f"### Top {top_n} Songs Similar to {selected_song}:")
    similarity_scores = compute_similarity(df, song_lyrics)
    df['similarity'] = similarity_scores
    df = df[df['Song Title'] != selected_song]
    recommended_songs = df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL', 'Media']]

# Main app
def main():
    st.title("Emotion-based and Lyric Similarity Song Recommender")
    
    # Add custom CSS for background
    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://wallpapercave.com/wp/wp11163687.jpg');
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Download the dataset
    df = download_data_from_drive()

    # Drop duplicate entries
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')

    # Convert 'Release Date' to datetime
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Ensure emotions are detected for songs before recommendations
    if 'Emotions' not in df.columns or df['Emotions'].isnull().all():
        df = detect_emotions_for_songs(df)

    # Search bar for song or artist
    search_term = st.text_input("Enter a Song Name or Artist").strip()

    if search_term:
        filtered_songs = df[
            (df['Song Title'].str.contains(search_term, case=False, na=False)) |
            (df['Artist'].str.contains(search_term, case=False, na=False))
        ].sort_values(by='Release Date', ascending=False).reset_index(drop=True)
        
        if filtered_songs.empty:
            st.write("No songs found matching the search term.")
        else:
            st.write(f"### Search Results for: {search_term}")
            for idx, row in filtered_songs.iterrows():
                st.write(f"**{idx + 1}. {row['Song Title']}** by {row['Artist']}")
                st.write(f"Album: {row['Album']} | Release Date: {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
                youtube_url = extract_youtube_url(row['Media'])
                if youtube_url:
                    video_id = youtube_url.split('watch?v=')[-1]
                    st.write(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                st.write("---")

            song_list = filtered_songs['Song Title'].unique()
            selected_song = st.selectbox("Select a Song for Recommendations", song_list)

            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, selected_song)
                if not recommendations.empty:
                    st.write(f"### Recommended Songs Similar to {selected_song}")
                    for idx, row in recommendations.iterrows():
                        st.write(f"**{idx + 1}. {row['Song Title']}** by {row['Artist']}")
                        st.write(f"Similarity Score: {row['similarity']:.2f}")
                        youtube_url = extract_youtube_url(row['Media'])
                        if youtube_url:
                            video_id = youtube_url.split('watch?v=')[-1]
                            st.write(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                        st.write("---")

if __name__ == "__main__":
    main()
