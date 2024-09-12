import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

genre_keywords = {
    'Rock': ['rock', 'guitar', 'band', 'drums'],
    'Pop': ['love', 'dance', 'hit', 'baby'],
    'Jazz': ['jazz', 'swing', 'blues', 'saxophone'],
    'Country': ['country', 'truck', 'road', 'cowboy'],
    'Hip Hop': ['rap', 'hip', 'hop', 'beat', 'flow'],
    'Classical': ['symphony', 'orchestra', 'classical', 'concerto']
}

def predict_genre(row):
    for genre, keywords in genre_keywords.items():
        text = f"{row['Song Title']} {row['Lyrics']}"
        if any(keyword.lower() in str(text).lower() for keyword in keywords):
            return genre
    return 'Unknown'

def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotions(lyrics, emotion_model):
    max_length = 512
    truncated_lyrics = ' '.join(lyrics.split()[:max_length])
    emotions = emotion_model(truncated_lyrics)
    return emotions

@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    song_lyrics = song_data['Lyrics'].values[0]
    song_genre = song_data['Predicted Genre'].values[0]
    
    emotion_model = load_emotion_model()
    song_emotion = detect_emotions(song_lyrics, emotion_model)
    
    similarity_scores = compute_similarity(df, song_lyrics)
    
    df['similarity'] = similarity_scores
    recommended_songs = df[df['Predicted Genre'] == song_genre].sort_values(by='similarity', ascending=False).head(top_n)
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'Predicted Genre', 'similarity']]

def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Genre")
    df = download_data_from_drive()
    df['Predicted Genre'] = df.apply(predict_genre, axis=1)

    # Create a dropdown selection for songs instead of search term input
    song_list = df['Song Title'].unique()
    selected_song = st.selectbox("Select a Song", song_list)
    
    if selected_song:
        st.write(f"### Selected Song: {selected_song}")
        
        # Show the song details and lyrics
        selected_song_data = df[df['Song Title'] == selected_song].iloc[0]
        st.markdown(f"*Artist:* {selected_song_data['Artist']}")
        st.markdown(f"*Album:* {selected_song_data['Album']}")
        release_date = pd.to_datetime(selected_song_data['Release Date'], errors='coerce')
        if pd.notna(release_date):
            st.markdown(f"*Release Date:* {release_date.strftime('%Y-%m-%d')}")
        else:
            st.markdown("*Release Date:* Unknown")
        
        with st.expander("Show/Hide Lyrics"):
            st.write(selected_song_data['Lyrics'].strip())
        
        # Show recommendations based on the selected song
        recommendations = recommend_songs(df, selected_song)
        if not recommendations.empty:
            st.write(f"### Recommended Songs Similar to {selected_song}")
            st.write(recommendations)
        else:
            st.write("No recommendations found.")

if __name__ == '__main__':
    main()
