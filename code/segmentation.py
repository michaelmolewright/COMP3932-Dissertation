import networkx as nx
import helper as util
import numpy as np
from sklearn.cluster import KMeans

def u0_function(phi_2):
    '''
    Function that creates an initial Function u_0(x)
    returns a mean constrainted second eigenvectors that is segmented into -1,1
    '''
    m = np.mean(phi_2)
    phi_2 = phi_2 - m
    phi_2[phi_2<0] = -1
    phi_2[phi_2>0] = 1
    
    return phi_2

class segment:
    """
    Creates an object that can segment a given graph

    Segmentation Algorithms:
    --Fielder Method
    --Perona + Freeman Method
    --Convex Splitting Scheme for GL
    """

    def setup(self, graph, laptype="normalized"):
        """
        Creates the Laplacian and associated eigenvalues and eigenvectors
        """
        if laptype == "normalized":
            self.laplacian = nx.normalized_laplacian_matrix(graph)
            eigvalues, eigVectors = np.linalg.eig(self.laplacian.A)
            self.eigens = util.sortEigens(eigvalues, eigVectors)
        elif laptype == "standard":
            self.laplacian = nx.laplacian_matrix(graph)
            eigvalues, eigVectors = np.linalg.eig(self.laplacian.A)
            self.eigens = util.sortEigens(eigvalues, eigVectors)
        else:
            print("that is not a valid laplacian type")

    def fielder_method(self):
        """
        Function that implements the Fielder Method
        Returns binary labels for the nodes of the graph
        """
        φ_2 = np.asarray(self.eigens["vectors"][1])
        kmeans = KMeans(n_clusters=2, n_init=10)

        kmeans.fit(φ_2.reshape(-1, 1))
        
        return kmeans.labels_

    def perona_freeman_method(self, k):
        """
        Function that implements the Perona Freeman Method
        Returns binary labels for the nodes of the graph
        """
        X = np.asarray(self.eigens["vectors"][:k])
        X = X.T
        #new_x = []
        #for i in X:
            #new_x.append(np.mean(i))
        #X = []
        '''
        for i in range(1, k+1):
            for j, value in enumerate(self.eigens["vectors"][i]):
                if i == 1:
                    X.append([value])
                else:
                    X[j].append(value)
        '''
        #new_x = np.asarray(new_x)
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(X)

        return kmeans.labels_
    
    
    def ginzburg_landau_segmentation_method(self, dt, c, epsilon, iterations):
        """ 
        Method that implemments the convex splitting scheme to minimize the GL Functional
        -----------
        dt : float - controls the time stepping of this algorithm
        c : float - convexity parameter
        epsilon : float - interface scaling value 
        iterations : maximum number of iterations of this algorithm
        """
        
        u_init = u0_function(np.asarray(self.eigens["vectors"][1]))

        #ALTERNATE initial function if needed
        #u_init = np.asarray(self.eigens["vectors"][1])
        #u_init = np.random.uniform(low = -1, high = 1, size = len(self.eigens["vectors"]))

        phi = np.asarray(self.eigens["vectors"])
        
        #initializing variables
        a_k = np.dot(u_init, phi.T)
        b_k = np.dot(np.power(u_init,3), phi.T)
        D_k = np.asarray(self.eigens["values"])
        D_k = 1 + (dt*(epsilon*D_k + c))
        d_k = np.zeros(len(D_k))
        temp = d_k + 1


        temp1 = (1+ (dt/epsilon) + (c*dt))
        u = u_init.copy()
        u_diff_old = 5000001
        u_diff_new = 5000000
        i = 0
        while (i<iterations) and (u_diff_old > u_diff_new): 
            
            u_old = u.copy()

            #update values for each iteration
            a_k = np.divide( (temp1*a_k) -  ((dt/epsilon)* b_k) - (dt * d_k), D_k )
            u = np.dot(a_k, phi)
            u = u - np.mean(u) #mean constraint
            b_k = np.dot(np.power(u,3), phi.T)
            d_k = np.dot( u - u_init , phi.T)

            #calculating convergence rate
            u_diff_old = u_diff_new
            u_diff_new = (abs(u-u_old)).sum()

            i += 1
        
        #segment into binary labels
        labels = u
        labels[labels<0] = 0
        labels[labels>0] = 1

        u_init = u

        return labels


    