# Gathers and preprocesses the data for the model and dashboard.

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import ast
from collections import Counter
from nrclex import NRCLex
import nltk
from retry_requests import retry
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from spotipy.oauth2 import SpotifyOAuth

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up Spotify API 
CLIENT_ID = "..."
CLIENT_SECRET = "..."
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Set up OpenAI API
oai_client = OpenAI(
    api_key="..."
)

# Get all playlists for a user
def get_all_playlists(user_id):
    playlists = sp.user_playlists(user_id)
    all_playlists = []
    
    while playlists:
        for playlist in playlists['items']:
            all_playlists.append({
                'id': playlist['id'],
                'name': playlist['name'],
                'description': playlist.get('description', ''),
                'public': playlist['public'],
                'collaborative': playlist['collaborative'],
                'total_tracks': playlist['tracks']['total']
            })
        playlists = sp.next(playlists) if playlists['next'] else None

    return pd.DataFrame(all_playlists)

# Get all tracks from a playlist
def get_tracks_from_playlist(playlist_id):
    """
    Returns a list of dictionaries containing track details.
    """
    tracks = []
    results = sp.playlist_tracks(playlist_id)
    
    while results:
        for item in results['items']:
            track = item.get('track')
            if not track:
                continue  # Skips if track was deleted

            tracks.append({
                'playlist_id': playlist_id,
                'track_id': track['id'],
                'track_name': track['name'],
                'track_duration_ms': track['duration_ms'],
                'album_id': track['album']['id'],
                'album_name': track['album']['name'],
                'album_release_date': track['album']['release_date'],
                'artist_ids': [artist['id'] for artist in track['artists']],
                'artist_names': [artist['name'] for artist in track['artists']],
                'added_at': item.get('added_at')
            })
        results = sp.next(results) if results['next'] else None
    
    return tracks


# Fetch lyrics for a track
def fetch_lyrics(track_name, artist_name):
    if isinstance(artist_name, list):
        artist_name = ", ".join(artist_name)
  
    track_name_encoded = requests.utils.quote(track_name)
    artist_name_encoded = requests.utils.quote(artist_name)
    lyrist_url = f'https://lyrist.vercel.app/api/{track_name_encoded}/{artist_name_encoded}'

    try:
        response = requests.get(lyrist_url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        return {'error': f'HTTP error occurred: {http_err}'}
    except Exception as err:
        return {'error': f'An error occurred: {err}'}


# Process lyrics and update the dataframe
# This function is meant to be called multiple times to avoid API rate limits.
def process_lyrics(input_csv, output_csv, max_requests=300):
    # Load existing output CSV if it exists
    try:
        existing_df = pd.read_csv(output_csv)
    except FileNotFoundError:
        df = pd.read_csv(f'data/{input_csv}')
        existing_df = pd.DataFrame(columns=df.columns.to_list() + ['lyrics'])

    # Remove duplicate songs to minimize API calls
    unique_df = df.drop_duplicates(subset=['track_name', 'artist_names'])

    # Keep only unprocessed rows
    processed_tracks = set(zip(existing_df['track_name'], existing_df['artist_names']))
    remaining_df = unique_df[~unique_df.apply(lambda row: (row['track_name'], row['artist_names']) in processed_tracks, axis=1)]
    processed_count = 0
    for _, row in remaining_df.iterrows():
        if processed_count >= max_requests:
            break
        artist = row['artist_names']

        # Parse artist string (it's a fake list)
        if isinstance(artist, str) and artist.startswith('[') and artist.endswith(']'):
            try:
                artist_list = ast.literal_eval(artist)
                if isinstance(artist_list, list):
                    artist = ", ".join(artist_list)
            except Exception:
                pass

        lyrics_data = fetch_lyrics(row['track_name'], artist)

        if 'error' not in lyrics_data and 'lyrics' in lyrics_data:
            lyrics = lyrics_data['lyrics']
        else:
            lyrics = None  # If there is an error or no lyrics

        new_row = {**row, 'lyrics': lyrics}
        existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        processed_count += 1
        print(f"Processed: {row['track_name']} - {artist}")

    # Save progress to CSV
    existing_df.to_csv(f'data/{output_csv}', index=False)
    print(f"Saved {processed_count} new entries to {output_csv}")

# Get all playlists and tracks for a user
def save_spotify_data(user_id = 'ethanmccoy-us'):
    user_playlists = get_all_playlists(user_id)

    # Get all tracks of the user's playlists
    all_tracks = []
    for playlist_id in user_playlists['id']:
        all_tracks.extend(get_tracks_from_playlist(playlist_id))
    user_tracks = pd.DataFrame(all_tracks)

    # Combine and save the playlists and tracks into one dataframe
    combined_df = user_tracks.merge(user_playlists, left_on='playlist_id', right_on='id', how='left')
    combined_df = combined_df.drop(columns=['id'])
    combined_df.to_csv('data/combined.csv', index=False)

# Compute emotion scores for a given text using the NRCLex lexicon.
def compute_emotion_scores(text):
    # Return zero scores if missing or empty lyrics
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {
            'fear': 0.0, 
            'anger': 0.0, 
            'anticipation': 0.0, 
            'trust': 0.0, 
            'surprise': 0.0, 
            'positive': 0.0, 
            'negative': 0.0, 
            'sadness': 0.0, 
            'disgust': 0.0, 
            'joy': 0.0
        }

    # Add emotions with NRCLex
    emotion = NRCLex(text)
    scores = emotion.affect_frequencies
    for e in ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']:
        if e not in scores:
            scores[e] = 0.0

    return scores

# Apply the compute_emotion_scores function to each row in the dataframe
def add_emotion_features(input_csv, lyrics_col='lyrics'):
    df = pd.read_csv(f'data/{input_csv}')

    # Emotion categories from NRCLex
    emotion_categories = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
    for e in emotion_categories:
        df[e] = 0.0

    # Emotion scores for each lyric
    for i, row in df.iterrows():
        text = row[lyrics_col]
        scores = compute_emotion_scores(text)
        for e in emotion_categories:
            df.at[i, e] = scores[e]

    df.to_csv('data/lyrics_with_emotions.csv', index=False)
    return df


def gpt_rate_lyrics(input_csv='data/lyrics_with_emotions.csv', output_csv='data/lyrics_with_ratings.csv', lyrics_col='lyrics', max_requests=5):
    class LyricsRating(BaseModel):
        Happy: int
        Happy_explanation: str
        Sad: int
        Sad_explanation: str
        Fear: int
        Fear_explanation: str
        Anger: int
        Anger_explanation: str
        Hope: int
        Hope_explanation: str
        Happy_Love: int
        Happy_Love_explanation: str
        Sad_Love: int
        Sad_Love_explanation: str
        Growth: int
        Growth_explanation: str
        Spirituality: int
        Spirituality_explanation: str
        Materialism: int
        Materialism_explanation: str
        Emotion: int
        Emotion_explanation: str
        Humor: int
        Humor_explanation: str
        Ambition: int
        Ambition_explanation: str
        Empowerment: int
        Empowerment_explanation: str
        Self_Reflection: int
        Self_Reflection_explanation: str
        Creativity: int
        Creativity_explanation: str
        Optimism: int
        Optimism_explanation: str
        Pessimism: int
        Pessimism_explanation: str
        Loneliness: int
        Loneliness_explanation: str
        Intensity: int
        Intensity_explanation: str
        Metaphorical: int
        Metaphorical_explanation: str

    # Load existing output CSV if it exists
    df = pd.read_csv(input_csv)
    try:
        existing_df = pd.read_csv(output_csv)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=df.columns.to_list() + ['ratings'])

    # Remove duplicate songs to minimize API calls
    unique_df = df.drop_duplicates(subset=['track_name', 'artist_names'])

    # Keep only rows that are not yet processed
    processed_tracks = set(zip(existing_df['track_name'], existing_df['artist_names']))
    remaining_df = unique_df[~unique_df.apply(lambda row: (row['track_name'], row['artist_names']) in processed_tracks, axis=1)]
    
    def process_row(row):
        text = row[lyrics_col]
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return None  # Skip if no valid lyrics

        # Call GPT-4o mini to get ratings
        response = oai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Rate these song lyrics on a scale of 1-10 for the following features: Happy, Sad Fear, Anger, Hope, Happy Love, Sad Love, Growth, Spirituality, Materialism, Emotion, Humor, Ambition, Empowerment, Self-Reflection, Creativity, Optimism, Pessimism, Loneliness, Intensity, Metaphorical. Lyrics: {text}"}
            ],
            response_format=LyricsRating,
            max_tokens=2000,
            temperature=0.7
        )

        # Parse into a LyricsRating object
        try:
            response_dict = json.loads(response.choices[0].message.content)
            ratings = LyricsRating.model_validate(response_dict)
            ratings_dict = {f'GPT_{feature}': getattr(ratings, feature) for feature in LyricsRating.model_fields.keys()}
            return {**row, **ratings_dict}
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Error parsing response: {e}")
            return None

    processed_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, row): row for _, row in remaining_df.iterrows()}
        for future in as_completed(futures):
            if processed_count >= max_requests:
                break  # Exit the loop once max_requests is reached

            result = future.result()
            if result:
                existing_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
                processed_count += 1
                print(f"Processed {processed_count} of {max_requests} ({processed_count/total_to_process*100:.2f}%): {result['track_name']} - {result['artist_names']}")

    # Save progress to CSV
    existing_df.to_csv(output_csv, index=False)
    print(f"Saved {processed_count} new entries to {output_csv}")



def merge_lyrics_emotions(combined_csv='data/combined.csv', lyrics_emotions_csv='data/lyrics_with_emotions.csv', output_csv='data/merged.csv'):
    """
    Merges lyrics and emotion data into the combined Spotify data, removing duplicate columns.
    Also calculates average emotion scores for each album and each lyric.
    """
    # Load the combined data and the lyrics with emotions data
    combined_df = pd.read_csv(combined_csv)
    lyrics_emotions_df = pd.read_csv(lyrics_emotions_csv)

    # Merge the dataframes on track_name and artist_names
    merged_df = combined_df.merge(
        lyrics_emotions_df,
        on=['track_name', 'artist_names'],
        how='left',
        suffixes=('', '_drop')  # Use a suffix that indicates columns to drop
    )

    # Drop the duplicate columns that have the '_drop' suffix
    columns_to_drop = [col for col in merged_df.columns if col.endswith('_drop')]
    merged_df.drop(columns=columns_to_drop, inplace=True)

    # Calculate average emotion scores for each album/lyric
    emotion_columns = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
    album_avg_emotions = merged_df.groupby('album_name')[emotion_columns].mean().reset_index()
    album_avg_emotions.columns = ['album_name'] + [f'album_avg_{col}' for col in emotion_columns]
    merged_df = merged_df.merge(album_avg_emotions, on='album_name', how='left')

    # Convert 'added_at' to datetime, extract hour and weekday
    merged_df['added_at'] = pd.to_datetime(merged_df['added_at'])
    merged_df['hour_added'] = merged_df['added_at'].dt.hour
    merged_df['weekday_added'] = merged_df['added_at'].dt.day_name()
    merged_df['month_added'] = merged_df['added_at'].dt.month_name()

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to {output_csv}")


# Removes all entries from the CSV where the playlist name contains 'top'.
def clean_csv_by_playlist_name(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    cleaned_df = df[~df['name'].str.contains('top', case=False, na=False)]
    
    cleaned_count = len(cleaned_df)
    removed_count = initial_count - cleaned_count
    
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Removed {removed_count} entries. Cleaned data saved to {output_csv}")

# Set up Spotify API with OAuth2
scope = "playlist-modify-public"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
client_id = "...",
client_secret = "...",
    scope=scope,
    redirect_uri='http://localhost:8080' 
))

# Creates Spotify playlists for each cluster
def create_cluster_playlists(cluster_data, user_id):
    for cluster_num in sorted(cluster_data.keys(), key=int):  # process in order
        cluster_info = cluster_data[cluster_num]
        playlist_name = f'Cluster {cluster_num}'
        
        # Make description 
        means_description = ', '.join([f"{key}: {value:.2f}" for key, value in cluster_info['means'].items()])
        description = f'Playlist for Cluster {cluster_num}. Means: {means_description}'
        
        # Create playlist
        playlist = sp.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=True,
            collaborative=False,
            description=description
        )
        playlist_id = playlist['id']
        
        song_ids = cluster_info['song_ids']
        
        # Add songs to playlist in batches of 100 (Spotify API limit)
        for i in range(0, len(song_ids), 100):
            sp.playlist_add_items(playlist_id, song_ids[i:i+100])


# Step 1:
# Run save_spotify_data() once to get combined.csv

# Step 2:
# Run process_lyrics(input_csv, output_csv) as many times as needed to process everything
# (Currently doing this manually to prevent shutting down the API)

# Step 3:
# Run add_emotion_features()

# Step 4: 
# gpt_rate_lyrics(input_csv='data/lyrics_with_emotions.csv', output_csv='data/lyrics_with_ratings.csv', max_requests=1000)

# Step 5: 
# Run merge_lyrics_emotions()
# merge_lyrics_emotions(combined_csv='data/combined.csv', lyrics_emotions_csv='data/lyrics_with_ratings.csv', output_csv='data/merged.csv')

# Step 6: 
# clean_csv_by_playlist_name('data/merged.csv', 'data/merged_cleaned.csv')

# Step 7: 
# Load cluster data from JSON file
# with open('cluster_data.json', 'r') as file:
#     cluster_data = json.load(file)
# create_cluster_playlists(cluster_data, user_id='ethanmccoy-us')

# Plots each training step
def plot_processing_steps(input_csv='data/merged.csv', min_samples_per_class=5):
    df = pd.read_csv(input_csv)
    initial_count = len(df)
    
    # Step 1: Remove songs without lyrics
    df_with_lyrics = df.dropna(subset=['lyrics'])
    lyrics_count = len(df_with_lyrics)
    
    # Step 2: Filter for songs including specific keywords
    artist_keywords = ['bladee', 'ecco', 'lean', 'thaiboy']
    df_filtered = df_with_lyrics[df_with_lyrics['artist_names'].str.contains('|'.join(artist_keywords), case=False, na=False)]
    filtered_count = len(df_filtered)
    
    # Step 3: Remove songs with less than min_samples_per_class samples
    class_counts = df_filtered['track_name'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_final = df_filtered[df_filtered['track_name'].isin(valid_classes)]
    final_count = len(df_final)
    
    # Plot
    steps = ['Initial', 'With Lyrics', 'Filtered for Artists', 'With 4+ Samples']
    counts = [initial_count, lyrics_count, filtered_count, final_count]
    plt.figure(figsize=(10, 6))
    plt.bar(steps, counts, color=['blue', 'orange', 'green', 'red'])
    plt.title('Number of Rows After Each Processing Step')
    plt.xlabel('Processing Step')
    plt.ylabel('Number of Rows')
    plt.show()

# plot_processing_steps()