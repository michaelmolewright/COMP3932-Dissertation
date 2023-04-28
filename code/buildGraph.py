import networkx as nx
import numpy as np
import math

def dist(A, B):
    '''
    Calculates the Euclidiean distance between two vectors (arrays)
    '''
    total = 0
    for i in range(len(A)):
        total += (A[i] - B[i])**2
    return total**0.5

def cosine(A,B):
    '''
    Compte the cosine value between two vectors, will be used as a similarity function
    '''
    part1 = 0
    part2 = 0
    part3 = 0

    for i in range(len(A)):
        part1 += A[i] * B[i]
        part2 += A[i]**2
        part3 += B[i]**2

    cos = part1 / ( part2**0.5 * part3**0.5 )
    return cos 

class graphBuilder:
    """
    Creates an object that can create a networkX graph from data inputed into the object

    Similarity funcitons (norm):
    --Gaussian Function
    --Local Scaling Function
    --Cosine Function
    --Gaussian Function (image)
    --Local Scaling Function (image)
    --Cosine Function (image)

    """

    def setup(self, data):
        '''
        inputs:
            data - This is an array that contains the feature vectors for the associated data
        '''
        self.data = data
        G = nx.Graph()
        for i in range(len(data)):
            G.add_node(i)
        self.graph = G


    def gaussian(self, τ):
        '''
        Creates a fully connected graph with edge weights calculated using the gaussian function

        τ - denotes a variable that can be used to adjust the outcome of the weight function
        '''
        self.graph = nx.create_empty_copy(self.graph)

        for i in range(0,len(list(self.graph.nodes))):
            for j in range(i,len(list(self.graph.nodes))):
                if j != i:
                    d = dist(self.data[i], self.data[j])
                    self.graph.add_edge(i, j, weight= 2.781 ** ( -((d ** 2)) / 2 * τ**2 ))

    def local_scaling(self, M):
        '''
        Creates a connected graph with edge weights calculated using the gaussian function but with local scaling weights 

        M - the desired number of nearest neighbours
        '''
        self.graph = nx.create_empty_copy(self.graph)

        all_distances = []

        for i in range(0,len(list(self.graph.nodes))):
            init_pos = self.data[i]
            distances = []
            
            for n in list(self.graph.nodes):
                distances.append(dist(self.data[i], self.data[n]))
            
            distances.sort()
            all_distances.append([i, distances[M]] )

        for i in range(0,len(list(self.graph.nodes))):
            for n in range(i+1,len(list(self.graph.nodes))):
                if n != i:
                    d = dist(self.data[i], self.data[n])
                    if (d <= all_distances[i][1]):
                        self.graph.add_edge( n, i, weight= 2.781 ** ( -((d ** 2)) / (all_distances[n][1] * all_distances[i][1] )) )
        
    def cosine(self):
        '''
        Creates a fully connect graph using the consine function as
        the similarity between the nodes in the graph
        '''
        self.graph = nx.create_empty_copy(self.graph)

        for i in range(0,len(list(self.graph.nodes))):
            for j in range(i+1,len(list(self.graph.nodes))):
                self.graph.add_edge( i, j, weight = math.exp( cosine(self.data[i],self.data[j]) ) ) #math.exp(cosine(self.data[i],self.data[j]))

    def gaussian_image(self, w, r, sig_gaus, sig_dist, gtype="intensity"):
        '''
        Creates a connected graph using different 
        '''
        self.graph = nx.create_empty_copy(self.graph)
        for i, RGB1 in enumerate(self.data):
            for j, RGB2 in enumerate(self.data):
                if j > i:
                    x1 = i % w
                    y1 = i // w
                    x2 = j % w
                    y2= j // w

                    pixels_away = ( (x2-x1)**2 + (y2-y1)**2 )**0.5

                    if pixels_away < r:
                        if gtype == "intensity":
                            I1 = (RGB1[0] + RGB1[1] + RGB1[2]) /3
                            I2 = (RGB2[0] + RGB2[1] + RGB2[2]) /3

                            new_d = abs(I1 - I2) / 10 # has to be absolout to avoid complex eigens
                            distanceVal = (pixels_away ** 2) / sig_dist
                            colourVal = math.exp( -((new_d)) / sig_gaus )
                            yo = colourVal * distanceVal

                            self.graph.add_edge(i, j, weight = yo )
                        elif gtype == "colour":
                            new_d = dist(RGB1, RGB2) / 20
                            distanceVal = (pixels_away ** 2) / sig_dist
                            colourVal = math.exp( -((new_d)) / sig_gaus )
                            finalVal = colourVal * distanceVal

                            self.graph.add_edge(i, j, weight = finalVal )
                        elif gtype == "cosine":
                            new_d = cosine(RGB1, RGB2)
                            distanceVal = (pixels_away ** 2) / sig_dist
                            colourVal = math.exp( new_d / sig_gaus )
                            finalVal = colourVal * distanceVal
                            
                            self.graph.add_edge(i, j, weight = finalVal )
                        
                        else:
                            print("not a valid gtype -- try again")
                            return

    

