import networkx as nx
import helper as util
import numpy as np
from sklearn.cluster import KMeans

def u0_function(phi_2):
    m = np.mean(phi_2)
    phi_2 = phi_2 - m
    phi_2[phi_2<0] = -1
    phi_2[phi_2>0] = 1
    
    return phi_2

def _diffusion_step_eig(v,V,E,dt):
    """diffusion on graphs
    """
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
        φ_2 = np.asarray(self.eigens["vectors"][1])
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
    
    
    def ginzburg_landau_segmentation_method(self, dt, c, epsilon, iterations):

        u_init = u0_function(np.asarray(self.eigens["vectors"][1]))
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

            #updating values
            a_k = np.divide( (temp1*a_k) -  ((dt/epsilon)* b_k) - (dt * d_k), D_k )
            u = np.dot(a_k, phi)
            u = u - np.mean(u)
            b_k = np.dot(np.power(u,3), phi.T)
            #d_k = np.dot( u - u_init , phi.T)#np.dot( (u - u_init) , phi.T)
            #u = u - np.mean(u)
            #calculating convergence rate
            u_diff_old = u_diff_new
            u_diff_new = (abs(u-u_old)).sum()
            #print(u_diff_new)
            i += 1
        
        labels = u
        labels[labels<0] = 0
        labels[labels>0] = 1

        #kmeans = KMeans(n_clusters=2, n_init=10)
        #kmeans.fit(u.reshape(-1, 1))
        u_init = u

        return labels

    def gl_zero_means_eig(self,dt,eps, tol = 1e-5,Maxiter = 200, inner_step_count = 10): 
        """ The MBO scheme with a forced zero mean constraint. Valid only for binary classification. 
        Parameters
        -----------
        V : ndarray, shape (n_samples, Neig)
            collection of smallest eigenvectors
        E : ndarray, shape (n_samples, 1)
            collection of smallest eigenvalues
        tol : scalar, 
            stopping criterion for iteration
        Maxiter : int, 
            maximum number of iterations
        """
        V = np.asarray(self.eigens["vectors"][:20]).T
        E = np.asarray(self.eigens["values"][:20])
        E = E[:,np.newaxis]
        u_init = np.asarray(self.eigens["vectors"][1])
        i = 0
        u_new = u_init.copy()
        u_diff = 1

        while (i<Maxiter) and (u_diff > tol):
            u_old = u_new.copy()
            w = u_old.copy()
            for k in range(inner_step_count): # diffuse and threshold for a while
                v = _diffusion_step_eig(w,V,E,eps*dt)
                w = v-np.mean(v) # force the 0 mean
            u_new = _gl_forward_step(w,dt,eps)
            u_diff = (abs(u_new-u_old)).sum()

            i = i+1
        
        labels = u_new
        labels[labels<0] = 0
        labels[labels>0] = 1

        return labels


    