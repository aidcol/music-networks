import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import time
from tqdm import tqdm
import os

CLIENT_ID = '253ad1b3d7b9444ca2ff45020afc695e'
CLIENT_SECRET = '37e521e9913a418a95bc2193bc7c0d14'

sp = Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
))

file_path = '/Users/noah01px2019/Desktop/Spotify2/exact_overlaps.csv'
progress_file = '/Users/noah01px2019/Desktop/Spotify3/progress.csv'
output_file = '/Users/noah01px2019/Desktop/Spotify3/track_features_with_genres.csv'

data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
required_data = data[['Artist', 'Song']]

if os.path.exists(progress_file):
    progress_df = pd.read_csv(progress_file)
    processed_tracks = set(zip(progress_df['Artist'], progress_df['Song']))
else:
    processed_tracks = set()

if os.path.exists(output_file):
    features_df = pd.read_csv(output_file)
    features_list = features_df.to_dict(orient='records')
else:
    features_list = []

def get_track_features(artist, song, retries=5):
    query = f"artist:{artist} track:{song}"
    attempt = 0
    while attempt < retries:
        try:
            results = sp.search(q=query, type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                artist_id = track['artists'][0]['id']
                
                features = sp.audio_features([track_id])[0]

                artist_info = sp.artist(artist_id)
                genres = artist_info.get('genres', [])
                
                return {
                    'Danceability': features['danceability'],
                    'Energy': features['energy'],
                    'Key': features['key'],
                    'Loudness': features['loudness'],
                    'Mode': features['mode'],
                    'Speechiness': features['speechiness'],
                    'Acousticness': features['acousticness'],
                    'Instrumentalness': features['instrumentalness'],
                    'Liveness': features['liveness'],
                    'Valence': features['valence'],
                    'Tempo': features['tempo'],
                    'Genres': ', '.join(genres) 
                }
        except Exception as e:
            print(f"Error fetching data for {artist} - {song}: {e}")
            break
        time.sleep(2 ** attempt)  #delay to help rate limiting (it didn't seem to solve anything :(
        attempt += 1
    
    return None

requests_per_second = 40 #Spotify API limit is 50 per sec
delay = 1.0 / requests_per_second

for _, row in tqdm(required_data.iterrows(), total=len(required_data), desc="Fetching Spotify Data"):
    artist, song = row['Artist'], row['Song']
    
    if (artist, song) in processed_tracks:
        continue
    
    features = get_track_features(artist, song)
    if features:
        features['Artist'] = artist
        features['Song'] = song
        features_list.append(features)
        processed_tracks.add((artist, song))  

    if len(processed_tracks) % 10 == 0:  #Saving progress every 10 tracks
        pd.DataFrame(features_list).to_csv(output_file, index=False)
        pd.DataFrame(list(processed_tracks), columns=['Artist', 'Song']).to_csv(progress_file, index=False)
    
    time.sleep(delay)

pd.DataFrame(features_list).to_csv(output_file, index=False)
pd.DataFrame(list(processed_tracks), columns=['Artist', 'Song']).to_csv(progress_file, index=False)
print(f"Data saved to {output_file}. Progress saved to {progress_file}.")
