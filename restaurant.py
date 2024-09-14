import pandas as pd
import streamlit as st
from transformers import pipeline, AutoTokenizer

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
        inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, max_length=max_length)
        emotions = emotion_model(lyrics[:tokenizer.model_max_length])
        return emotions
    except Exception as e:
        st.write(f"Error in emotion detection for lyrics: {lyrics[:100]}...: {e}")
        return []

# Detect emotions for all songs in the dataset
def detect_emotions_for_songs(df):
    emotion_model, tokenizer = load_emotion_model()
    
    # Handle missing or invalid lyrics
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    
    df['Emotions'] = df['Lyrics'].apply(lambda lyrics: detect_emotions(lyrics, emotion_model, tokenizer))
    return df

# Main function
def main():
    # Simulate a dataset for the example
    data = {
        'Song Title': ['Song 1', 'Song 2', 'Song 3'],
        'Lyrics': ['This is a test lyric', '', 'Another lyric']
    }
    df = pd.DataFrame(data)

    # Detect emotions for songs
    df = detect_emotions_for_songs(df)
    
    # Show the dataframe with emotions
    st.write(df)

if __name__ == '__main__':
    main()
