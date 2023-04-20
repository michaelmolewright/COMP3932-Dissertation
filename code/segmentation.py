import networkx as nx
import helper as util
import numpy as np
from sklearn.cluster import KMeans


def _diffusion_step_eig(v,V,E,dt):

    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()

def _gl_forward_step(u_old,dt,eps):
    v = u_old-dt/eps*(np.power(u_old,3)-u_old) #double well explicit step
    return v


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
        φ_2 = self.eigens["vectors"][1]
        kmeans = KMeans(n_clusters=2, n_init=10)

        kmeans.fit(φ_2.reshape(-1, 1))
        
        return kmeans.labels_

    def perona_freeman_method(self, k):
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
    
    def gl_method(self, dt, c, epsilon, iterations, k):
        
        u_init = self.eigens["vectors"][1]
        
        Eval = np.asarray(self.eigens["values"][:k])
        Evec = np.asarray(self.eigens["vectors"][:k])
        eigenVectors = Evec.T
        eigenValues = Eval[:,np.newaxis]


        i = 0
        u_new = u_init.copy()
        u_diff = 1
        tol = 0.00001

        while (i<iterations) and (u_diff > tol):
            u_old = u_new.copy()
            w = u_old.copy()
            for k in range(10):
                v = _diffusion_step_eig(w,eigenVectors,eigenValues,epsilon*dt)
                w = v-np.mean(v) 
            u_new = _gl_forward_step(w,dt, epsilon)
            u_diff = (abs(u_new-u_old)).sum()

            i = i+1

        labels = u_new
        labels[labels<0] = 0
        labels[labels>0] = 1

        return labels

    