import streamlit as st
import requests
import pandas as pd
import gdown

# Function to download the CSV from Google Drive
@st.cache
def download_data_from_drive():
    # Google Drive link for the dataset (convert to direct download link)
    url = 'https://drive.google.com/uc?id=1Tc3Hequ5jVjamAfuPhpBv8JvsOp7LSJY'
    output = 'restaurant_reviews.csv'
    
    # Download the file without printing progress (quiet=True)
    gdown.download(url, output, quiet=True)
    
    # Load the dataset
    return pd.read_csv(output)

# Load the dataset of restaurant reviews
reviews_df = download_data_from_drive()

# Geoapify API key
GEOAPIFY_API_KEY = "1b8f2a07690b4cde9b94e68770914821"

# JavaScript code to get browser's geolocation
def get_geolocation_script():
    geolocation_code = """
        <script>
        function getLocation() {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const latitude = position.coords.latitude;
                    const longitude = position.coords.longitude;
                    const coords = latitude + "," + longitude;
                    // Send data to Streamlit via hidden input field
                    document.getElementById("geo-input").value = coords;
                    document.getElementById("geo-input").dispatchEvent(new Event('change'));
                }
            );
        }
        </script>
    """
    return geolocation_code

# Inject JavaScript into Streamlit
st.markdown(get_geolocation_script(), unsafe_allow_html=True)

# Title
st.title("Restaurant Recommendation System")

# Coordinates input with hover effect to show 'Use my current location' button
coords = st.text_input("Enter your coordinates (latitude,longitude):", key="geo-input")

# Display 'Use my current location' button below the input
st.markdown("""
    <style>
    #geo-input:hover + button {
        display: block;
    }
    button#geo-location-btn {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Add the 'Use my current location' button
if st.button('Use my current location', key='geo-location-btn', help='Click to auto-fill coordinates with your current location'):
    st.write('<script>getLocation();</script>', unsafe_allow_html=True)

# Check if coordinates are available
if coords:
    lat, lon = map(float, coords.split(","))
    st.write(f"Detected Location: (Latitude: {lat}, Longitude: {lon})")

    # Use Geoapify Places API to fetch restaurant recommendations
    def get_restaurant_recommendations(lat, lon):
        url = f"https://api.geoapify.com/v2/places?categories=catering.restaurant&filter=circle:{lon},{lat},5000&limit=10&apiKey={GEOAPIFY_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            restaurants = data["features"]
            restaurant_list = [
                {
                    "name": place["properties"].get("name", "Unknown name"),
                    "address": place["properties"].get("formatted", "No address available"),
                    "category": place["properties"]["categories"][0]
                }
                for place in restaurants
            ]
            return restaurant_list
        else:
            st.error("Failed to retrieve restaurant data.")
            return []

    # Get restaurant recommendations based on the exact location
    st.header("Nearby Restaurant Recommendations:")
    restaurants = get_restaurant_recommendations(lat, lon)

    if restaurants:
        for idx, restaurant in enumerate(restaurants):
            # Initialize session state to track whether to show details or not
            if f"show_details_{idx}" not in st.session_state:
                st.session_state[f"show_details_{idx}"] = False

            # Button to toggle showing/hiding details
            if st.button(f"{restaurant['name']}", key=f"restaurant_{idx}"):
                # Toggle the visibility state
                st.session_state[f"show_details_{idx}"] = not st.session_state[f"show_details_{idx}"]

            # Conditionally display restaurant details if the button is pressed (toggled on)
            if st.session_state[f"show_details_{idx}"]:
                st.write(f"**{restaurant['name']}**")
                st.write(f"Address: {restaurant['address']}")
                st.write(f"Category: {restaurant['category']}")
                st.write("---")

                # Extract reviews for the recommended restaurant
                restaurant_reviews = reviews_df[reviews_df["Restaurant"].str.contains(restaurant['name'], case=False, na=False)]
                
                if not restaurant_reviews.empty:
                    st.write("**Reviews:**")
                    for _, review_row in restaurant_reviews.iterrows():
                        st.write(f"- {review_row['Review']} (Rating: {review_row['Rating']})")
                else:
                    st.write("No reviews found.")
                st.write("---")
    else:
        st.write("No restaurants found nearby.")
else:
    st.write("Waiting for coordinates...")
