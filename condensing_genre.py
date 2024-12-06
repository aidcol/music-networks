import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


class Genre_Condenser():
    def __init__(self):
        self.labels = ["pop", "rock", "r&b", "rap", "jazz", "classical", "country", "latin", "k-pop", "edm"]
        self.edm_labels = ["house", "electro"]
        self.rap_labels = ["hip hop"]
        self.rb_labels = ["soul"]
        self.latin_labels = ["mexican", "puerto rican", "ranchera"]
        self.rock_labels = ["metal"]
        self.counts = np.zeros(len(self.labels))

    def run_labels(self, input_genres):
        for i in range(len(self.labels)):
            g_count = input_genres.count(self.labels[i])
            self.counts[i] = g_count

    def add_extra(self, input_genres):
        edm_count = 0
        rap_count = 0
        rb_count = 0
        latin_count = 0
        rock_count = 0
        for edm_label in self.edm_labels:
            edm_count += input_genres.count(edm_label)
        for rap_label in self.rap_labels:
            rap_count += input_genres.count(rap_label)
        for rb_label in self.rb_labels:
            rb_count += input_genres.count(rb_label)
        for latin_label in self.latin_labels:
            latin_count += input_genres.count(latin_label)
        for rock_label in self.rock_labels:
            rock_count += input_genres.count(rock_label)
        self.counts[1] += rock_count
        self.counts[2] += rb_count
        self.counts[3] += rap_count
        self.counts[7] += latin_count
        self.counts[9] += edm_count
        
    def condense(self, input_genres):
        if type(input_genres) != str:
            return "other"
        self.run_labels(input_genres)
        self.add_extra(input_genres)
        #prioritize k-pop and latin (they would pop up for other genres)
        if self.counts[8] > 0:
            return self.labels[8]
        if self.counts[7] > 0:
            return self.labels[7]
        if (self.counts == np.zeros(len(self.counts))).all():
            return "other"
        index_max = np.argmax(self.counts)
        return self.labels[index_max] 


#open the graph
with open("overlap_graph_small.pkl", "rb") as f:
    G = pickle.load(f)

#get the features
df = pd.read_csv("track_features_with_genres.csv")

#condense the genre to a label
test1 = Genre_Condenser()
df["Genres"] = df["Genres"].apply(lambda x: test1.condense(x))

#save the colorings
genre_mapping_df = pd.DataFrame()
genre_mapping_df["Genre"] = df["Genres"]
genre_mapping_df["Node"] = df.apply(lambda row: (row['Song'], row['Artist']), axis=1)
genre_mapping_df = genre_mapping_df.set_index(["Node"])
genre_mapping_df.head()
genre_mapping_df.to_csv('genre_coloring.csv') 

#assign the genres to the nodes
genre_dict = genre_mapping_df.to_dict()
genre_dict = genre_dict['Genre']
nx.set_node_attributes(G, genre_dict, name="genre")
with open("playlist_graph_with_genre.pkl", "wb") as file:
    pickle.dump(G, file)
