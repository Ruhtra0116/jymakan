# Import necessary libraries
import pandas as pd
import streamlit as st
import gdown

# Function to download the CSV from Google Drive
@st.cache_data
def download_data_from_drive():
    # Google Drive link for the dataset (convert to direct download link)
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'  # Replace FILE_ID with the actual file ID
    output = 'songTest1.csv'  # Change to the desired output file name
    
    # Download the file without printing progress (quiet=True)
    gdown.download(url, output, quiet=True)
    
    # Load the dataset
    return pd.read_csv(output)

# Load the dataset of your CSV file
data_df = download_data_from_drive()

# Define a dictionary with genre keywords
genre_keywords = {
    'Rock': ['rock', 'guitar', 'band', 'drums'],
    'Pop': ['love', 'dance', 'hit', 'baby'],
    'Jazz': ['jazz', 'swing', 'blues', 'saxophone'],
    'Country': ['country', 'truck', 'road', 'cowboy'],
    'Hip Hop': ['rap', 'hip', 'hop', 'beat', 'flow'],
    'Classical': ['symphony', 'orchestra', 'classical', 'concerto']
}

# Function to predict genre based on keywords in song title or lyrics
def predict_genre(row):
    for genre, keywords in genre_keywords.items():
        text = f"{row['Song Title']} {row['Lyrics']}"  # Combine relevant text fields
        if any(keyword.lower() in str(text).lower() for keyword in keywords):
            return genre
    return 'Unknown'  # Default if no keywords are matched

# Apply the genre prediction to each row in the dataset
data_df['Predicted Genre'] = data_df.apply(predict_genre, axis=1)

# Add a sidebar for filtering songs by predicted genre
st.sidebar.header('Filter Songs by Predicted Genre')

# Get unique genres from the predicted genres column for the dropdown
unique_genres = data_df['Predicted Genre'].unique()
unique_genres = [genre for genre in unique_genres if genre != 'Unknown']  # Exclude 'Unknown' if desired

# Dropdown selection for genres
selected_genre = st.sidebar.selectbox('Select a Genre', options=['Select a genre'] + unique_genres)

# Check if a valid genre is selected
if selected_genre != 'Select a genre':
    # Filter songs based on the selected genre
    filtered_songs = data_df[data_df['Predicted Genre'] == selected_genre]

    # Sort the filtered songs by 'Release Date' in descending order
    filtered_songs['Release Date'] = pd.to_datetime(filtered_songs['Release Date'], errors='coerce')  # Convert to datetime
    filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

    # Display each song in a banner format with an expander to show/hide lyrics
    st.write(f"### Songs Filtered by Genre: {selected_genre}")
    for idx, row in filtered_songs.iterrows():
        with st.container():
            # Combine the song number and title into a single line
            st.markdown(f"*No. {idx + 1}: {row['Song Title']}*")
            st.markdown(f"*Artist:* {row['Artist']}")
            st.markdown(f"*Album:* {row['Album']}")
            st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            
            # Use expander to show/hide lyrics
            with st.expander("Show/Hide Lyrics"):
                st.write(row['Lyrics'].strip())  # Clean up the lyrics display
            st.markdown("---")  # Separator between songs
else:
    st.write("Please select a genre to display the songs.")
