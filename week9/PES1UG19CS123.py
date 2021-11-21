import numpy as np

class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def cdist(self,centroids,points): #find the distance between points and the centroid
        assigned_centroids = [None for i in range(len(points))] #the closests centroids

        for i,point in enumerate(points):
            min_dist = float("inf")
            for j,centroid in enumerate(centroids):
                curr_dist = np.linalg.norm(point-centroid)
                if curr_dist <= min_dist:
                    min_dist = curr_dist
                    assigned_centroids[i] = j
        return assigned_centroids

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        return self.cdist(self.centroids,data)

    def clusterToPoints(self,cluster_assign,data):
        hmap = {} #cluster with points

        for i in range(len(cluster_assign)):
            cluster = cluster_assign[i]
            if cluster in hmap:
                hmap[cluster].append(data[i])
            else:
                hmap[cluster] = [data[i]]
        return hmap

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
            cluster_assign: Cluster Assignment
        Change self.centroids
        """
        hmap = self.clusterToPoints(cluster_assgn,data)

        for k,v in hmap.items(): #k->cluster #,v->array of points of the cluster
            self.centroids[k] = np.mean(v,axis=0,dtype=np.float64)

    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        hmap = self.clusterToPoints(cluster_assign,data)

        J=0
        for i in range(self.n_cluster):
            centroid = self.centroids[i] #centroid of cluster i
            for point in hmap[i]:
                J+=(np.linalg.norm(point-centroid)**2) #sqr of norm
        return J
