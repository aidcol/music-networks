import re
import csv

def extract_song_artist(filename):
    #Removes "clip", "sample", and parentheses
    filename = re.sub(r'\(.*?\)', '', filename, flags=re.IGNORECASE)
    filename = re.sub(r'\bclip\b|\bsample\b', '', filename, flags=re.IGNORECASE)
    filename = filename.strip()

    if ' by ' in filename.lower():
        parts = re.split(r' by ', filename, flags=re.IGNORECASE)
        artist_name = parts[1].strip().replace('.mp3', '').replace('.ogg', '')
        song_title = parts[0].strip().replace('.mp3', '').replace('.ogg', '')
        return artist_name, song_title

    elif '-' in filename:
        parts = re.split(r'\s-\s', filename)
        if len(parts) == 2: 
            artist_name = parts[0].strip()
            song_title = parts[1].replace('.mp3', '').replace('.ogg', '').strip()
            return artist_name, song_title
        else:
            return "", filename.replace('.mp3', '').replace('.ogg', '').strip()

    else:
        song_title = filename.replace('.mp3', '').replace('.ogg', '').strip()
        return "", song_title


csv_path_wikipedia = '/Users/noah01px2019/Desktop/filtered_mc.csv'

song_data_wikipedia = []
with open(csv_path_wikipedia, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader) 
    for idx, row in enumerate(reader, start=2): 
        filename = row[0].strip()
        artist, song = extract_song_artist(filename)
        song_data_wikipedia.append([artist, song, filename, idx]) 

# Remove duplicates
song_data_wikipedia = [list(item) for item in set(tuple(item) for item in song_data_wikipedia)]

csv_path_spotify = '/Users/noah01px2019/Desktop/Spotify/spotify_dataset.csv'

songs_artists = []

with open(csv_path_spotify, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter=',', quotechar='"', escapechar='\\')

    reader.fieldnames = [field.strip().strip('"') for field in reader.fieldnames]

    for row in reader:
        artist = row['artistname'].strip() 
        song = row['trackname'].strip() 
        songs_artists.append((artist, song))


wikipedia_song_data_set = set((artist, song) for artist, song, _, _ in song_data_wikipedia)
songs_artists_set = set(songs_artists)

exact_overlap_set = wikipedia_song_data_set.intersection(songs_artists_set)

exact_overlap = [
    (artist, song, original_name, original_row)
    for artist, song, original_name, original_row in song_data_wikipedia
    if (artist, song) in exact_overlap_set
]

print(f"Number of exact overlapping artist-song pairs: {len(exact_overlap)}")

exact_overlap_path = '/Users/noah01px2019/Desktop/exact_overlaps.csv'

# Save exact overlaps to a CSV file
with open(exact_overlap_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Artist', 'Song', 'Original Name', 'Original Row'])  # Header row
    writer.writerows(exact_overlap)

print(f"Exact overlaps saved to {exact_overlap_path}")
