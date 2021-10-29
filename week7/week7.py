import numpy as np
from numpy.lib import type_check
from sklearn.tree import DecisionTreeClassifier
from numpy import log as ln
from numpy import exp as e

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = [] # a list to sotre all the stumps

    def fit(self, X, y): #called finally to train the adaboost algorithm
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y) #inital weights to all the samples are the same

        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2) #stump is a decision tree of depth one

            st.fit(X, y, sample_weights)
            y_pred = st.predict(X) #output of the decision tree

            self.stumps.append(st) #append the stump

            error = self.stump_error(y, y_pred, sample_weights=sample_weights) #calculate the error prouced by the stump

            alpha = self.compute_alpha(error) #weight of the stump in the final prediction
            self.alphas.append(alpha)

            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha) #normalized updated sample weight

        return self

    def stump_error(self, y, y_pred, sample_weights): #sum of the weights of misclassified records
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """

        m = len(y)
        total_err_weights=0
        for i in range(0,m):
            if y[i]!=y_pred[i]:
                total_err_weights+=sample_weights[i]
        return total_err_weights

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        alpha = 0.5*ln((1-error)/(error+eps))
        return alpha

    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """

        #the weights of the wrongly classified sample is increased
        factor_inc = e(alpha)
        factor_dec = e(-alpha)

        m = len(y)
        for i in range(0,m):
            if y[i]!=y_pred[i]:
                sample_weights[i]*=factor_inc
            else:
                sample_weights[i]*=factor_dec
        
        #normalize them
        normalizing_fac = 0
        for i in range(0,m):
            normalizing_fac+=sample_weights[i]
        
        for i in range(0,m):
            sample_weights[i]/=normalizing_fac

        return sample_weights

    def predict(self, X): #predict the output of the ADABOOST algorithm
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        #calculate the weigted average of the weak classifiers
        alpha_sum=0
        for alpha in self.alphas:
            alpha_sum+=alpha

        total=None
        for i,stump in enumerate(self.stumps):
            weighted_stump_pred = stump.predict(X)*self.alphas[i]
            if i==0: #first time
                total=weighted_stump_pred
            else:
                total+=weighted_stump_pred
        total/=alpha_sum
        weighted_avg_pred=np.floor(total)
        return weighted_avg_pred

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy