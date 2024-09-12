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
    
    similarity_scores = compute_similarity(df, song_lyrics)
    
    df['similarity'] = similarity_scores
    recommended_songs = df[df['Predicted Genre'] == song_genre].sort_values(by='similarity', ascending=False).head(top_n)
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'Predicted Genre', 'similarity']]

def show_song_details(row):
    st.write(f"### Song Title: {row['Song Title']}")
    st.write(f"Artist: {row['Artist']}")
    st.write(f"Album: {row['Album']}")
    release_date = pd.to_datetime(row['Release Date'], errors='coerce')
    if pd.notna(release_date):
        st.write(f"Release Date: {release_date.strftime('%Y-%m-%d')}")
    else:
        st.write("Release Date: Unknown")
    st.write(f"Predicted Genre: {row['Predicted Genre']}")
    st.write("### Lyrics:")
    st.write(row['Lyrics'])

def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Genre")
    df = download_data_from_drive()
    df['Predicted Genre'] = df.apply(predict_genre, axis=1)

    # Show some initial recommendations
    st.write("### Initial Song Recommendations")
    initial_recommendations = df.sample(5)
    for idx, row in initial_recommendations.iterrows():
        st.write(f"No. {idx + 1}: {row['Song Title']} by {row['Artist']}")

    # Search bar for song name
    search_term = st.text_input("Enter a Song Name to Search").strip()

    if search_term:
        filtered_songs = df[df['Song Title'].str.contains(search_term, case=False, na=False)]

        if filtered_songs.empty:
            st.write("No songs found matching the search term.")
        else:
            # Display the first matching song details
            song_details = filtered_songs.iloc[0]
            st.write("### Song Details:")
            show_song_details(song_details)

            # Recommend similar songs
            if st.button("Recommend Similar Songs"):
                recommendations = recommend_songs(df, song_details['Song Title'])
                st.write(f"### Recommended Songs Similar to {song_details['Song Title']}")
                st.write(recommendations)
    else:
        st.write("Please enter a song name to search.")

if __name__ == '__main__':
    main()
