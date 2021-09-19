import numpy as np
import math

from pandas.core.dtypes.missing import isna

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
    def minkowski_dist(self,vec1,vec2): 
        '''
        vec1 and vec2 are arrays with D elements(where D= #of dim)
        check if len(vec1)!=len(vec2)
        len(vec1)==0 || len(vec2)==0
        one_by_p is 1/0
        '''
        n=len(vec1)
        res=0.0

        for i in range(0,n):
            #print(f"Abs of {vec1[i]} and {vec2[i]} is {abs(vec1[i]-vec2[i])}")
            #print(f"Power of above abs is {math.pow(abs(vec1[i]-vec2[i]),self.p)}")
            res+=math.pow(abs(vec1[i]-vec2[i]),self.p)
        return math.pow(res,(1/self.p))

    
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
        speed up with numpy arrays and numpy functions
        try using np.array_equal for equality test
        '''
        dists=[]
        
        for c in x: #for each of the query_point in list of query_points
            temp=[]
            for instance in self.data: #calc dist b/w data points

                #ignore the query point if it is present in the dataset
                #print(f"Query point with D dims {c}")
                #print(f"The instace is {instance}")
                #print(f"Types {type(c)} vs {type(instance)}")
                n=len(c)
                m=len(instance)
                if n==m:
                    same=True
                    for i in range(n):
                        if c[i]!=instance[i]:
                            same=False
                    if same: continue

                temp.append(self.minkowski_dist(c,instance))
            dists.append(temp)
        # print("Distances of every query point with dataset is ")
        
        # for d in dists:
        #     print(d)

        return dists

    def euclids_dist(self,vec1,vec2):
        dists=0
        for i in range(len(vec1)):
            dists+=math.sqrt(math.pow(vec1[i],2)+math.pow(vec2[i],2))
        return dists

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
        #distance between every query point c in x to the entire dataset 
        #distances is a NxM matrix
        distances=self.find_distance(x) 
        knn_indxs=[]
        knn_dists=[]

        #return the k nearest neighbouring nodes of every query point
        for c_dist in distances:
            #sort the distances along with the index
            indxs=np.argsort(c_dist)

            #0th indx will be distance of c to itself
            knn_indxs.append(indxs[0:self.k_neigh])
            knn_dists.append([c_dist[i] for i in indxs][0:self.k_neigh])

        #print(f"The sorted k nearest neighbours for each query point are at distances")
        # for d in knn_dists:
        #     print(f"{d}")

        return knn_dists,knn_indxs

    """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
    """
    def predict(self, x):
        #k_nns is an list of k nearest neighbours of c
        #if weighted knn,then out of the selected k nearest neighbours,
        #add (1/distance) of c from pos class points in k neis and add (1/dist) of c from neg class points in neis
        #return the class with highest sum

        '''
        Edge cases:
        Try finding mode using numpy
        '''
        
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
                    k=f"{self.target[i]}"
                    if k not in mode:
                        mode[k]=1
                    else:
                        mode[k]+=1
                #print(f"Mode {mode}")
                for k,v in mode.items():
                    k=int(k)
                    if v>max_mode or v==max_mode and k<max_label:
                        max_label=k
                        max_mode=v
                pred.append(max_label)

                #print(f"Final mode {mode}")
        #print(f"Pred {pred}")
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
    data=[
        [100,120,12,6],
        [110,130,14,5],
        [120,110,11,7],
        [100,140,13,7],
        [115,140,11,6]
    ]

    target=[
        0,
        1,
        1,
        0,
        1
    ]
    target=np.asarray(target)
    data=np.asarray(data)

    query_points= [
        [100,135,12,8]
    ]
    query_targets=[
        0
    ]
    query_points=np.asarray(query_points)
    query_targets=np.array(query_targets)

    knn = KNN(3,False,2)
    knn.fit(data,target)
    acc= knn.evaluate(query_points,query_targets)
    print(f"The accuracy is {acc*100}%")


