from _typeshed import Self
import numpy as np
import math

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance

        x-> list of query points c each of which is D dim
        c-> query point
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted #is weighted knn?
        self.k_neigh = k_neigh #num of neighbours
        self.p = p #used in minkowski dist calc


    def fit(self, data, target): #lazy algo
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data #store the trainig data
        self.target = target.astype(np.int64) #stores the target for each of the data point

        return self
    
    # between two vectors of D dim each
    def minkowski_dist(self,vec1,vec2,one_by_p): 
        '''
        vec1 and vec2 are arrays with D elements(where D= #of dim)
        check if len(vec1)!=len(vec2)
        len(vec1)==0 || len(vec2)==0
        one_by_p is 1/0
        '''
        n=len(vec1)
        res=0

        for i in range(0,n):
            res+=math.pow(abs(vec1[i],vec2[i]),self.p)
        return math.pow(res,one_by_p)

    
    """
        Find the Minkowski distance b/w all the points in the train dataset(self.data) to x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float) ie an array of D dim points
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
    """
    def find_distance(self, x):
        '''
        Edge case:
        p=0
        speed up with numpy arrays?
        '''
        one_by_p= 1/float(self.p)
        dists=[]
        
        for c in x: #for each of the query_point in list of query_points
            temp=[]
            for instance in self.data: #calc dist b/w data points
                temp.append(self.minkowski_dist(c,instance,one_by_p))
            dists.append(temp)
        return dists


    def euclids_dist(self,vec1,vec2):
        dists=0
        for i in range(len(vec1)):
            dists+=math.sqrt(math.pow(vec1[i],2)+math.pow(vec2[i],2))
        return dists
        
    def calc_nn(self,c):
        dists=[]
        indxs=[]

        for i,instance in enumerate(self.data):
            dists.append(self.euclids_dist(c,instance))
            indxs.append(i)

        #sort both the arrays
        dists,indxs=zip(*sorted(zip(dists,indxs)))

        #the 0th index would be distance between c and itself
        return dists[1:self.k_neigh+1],indxs[1:self.k_neigh+1]

    """
        Find K nearest neighbours of each point in train dataset x 
        Note that the point itself "is not to be included " in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float) -> list of query points
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)

            returns the arrays : 
                neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
                (a list containing the distance between each of the query points out of N and k nearest neighbours in self.data)

                idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input
                {a list containing the index of each of the k nearest neighbours of every query point against the entire self.data}

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
    """
    def k_neighbours(self, x):
        kNN_dist_for_every_c=[]
        knn_indx_for_every_c=[]

        #go through every query point
        for c in x:
            one,two=self.calc_nn(c)
            #one-> list of dist of c from k nearest neighbours-> k dim vec
            #two-> list of indxs of the knns ->k dim vec

            kNN_dist_for_every_c.append(one)
            knn_indx_for_every_c.append(two)

        return kNN_dist_for_every_c,knn_indx_for_every_c

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        #k_nns is an list of k nearest neighbours of c
        #if weighted knn,then out of the selected k nearest neighbours,
        #add (1/distance) of c from pos class points in k neis and add (1/dist) of c from neg class points in neis
        #return the class with highest sum
        
        k_nei_dists,k_nei_indxs = self.k_neighbours(x) #n knn_indx and knn_dists 
        pred=[]

        for k_nei_dist,k_nei_indx in zip(k_nei_dists,k_nei_indxs): #go through each of the [k nei details]
            if self.weighted:
                cumu_wts={}
                for a_dist,a_indx in zip(k_nei_dist,k_nei_indx):
                    wt=1/float(a_dist)
                    cumu_wts[f"{self.target[a_indx]}"]+=wt
                
                #max weight ie inv of distance
                max_wt_node=float("-inf")
                max_wt=float("-inf")
                for k,v in cumu_wts:
                    k=int(k)
                    if v>max_wt or v==max_wt and k<max_wt_node:
                        max_wt=v
                        max_wt_node=k
                pred.append(max_wt_node)
                
            else:
                #go through the k neighbours and which class they belong to 
                #return the class with highest mode
                #in case of conflict lexi
                max_label=float("-inf")      
                max_mode=float("-inf")
                mode={}

                for i in k_nei_indx:
                    mode[f"{self.target[i]}"]+=1

                for k,v in mode:
                    k=int(k)
                    if v>max_mode or v==max_mode and k<max_label:
                        max_label=k
                        max_mode=v
                pred.append(max_label)
        return pred
                

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        preds=self.predict(x)
        correct=0
        n=len(preds)
        for i in range(n):
            if preds[i]==y[i]:
                correct+=1
        return correct/float(n)


if __name__=="__main__":
    knn_test = KNN()

