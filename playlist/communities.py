import networkx as nx
import numpy as np
import pickle
import json
import os


os.chdir(r'C:\Users\Dawso\Elec573\final_project')

class communities:
    def __init__(self, G):
        self.G = G
        self.communities(self.G)
    
    def communities(self,G):
        self.lou = nx.community.louvain_communities(G, seed =123)
        self.lpa = list(nx.community.asyn_lpa_communities(G, seed = 254))
        self.greedy = list(nx.community.greedy_modularity_communities(G))
        self.lou = [list(community) for community in self.lou]
        self.lpa = [list(community) for community in self.lpa]
        self.greedy = [list(community) for community in self.greedy]
    
    def write_community_files(self):
        path = os.getcwd() + '/communities'
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        with open("louvain.json", "w") as file:
            json.dump(self.lou, file)
        with open("lpa.json", "w") as file:
            json.dump(self.lpa, file)
        with open("greedy.json", "w") as file:
            json.dump(self.greedy, file)
        
        



with open("playlist_graph_with_genre.pkl", "rb") as file:
    G = pickle.load(file)


comms = communities(G)
comms.write_community_files()