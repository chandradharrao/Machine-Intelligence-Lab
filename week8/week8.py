from typing import Deque
import numpy as np
from collections import deque

'''
######CHECK############
optimization : use numpy array
multiply in the end
check for underflow
'''

class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix - transmission probabilities
        states: list of states - hidden states 
        emissions: list of observations - observable states
        B: Emmision probabilites 
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A #transmission probabilities
        self.B = B #emission probabilities
        self.states = states #list of the hidden states
        self.emissions = emissions #list of available observation states
        self.pi = pi #starting probabilties
        self.N = len(states) #number of hidden states
        self.M = len(emissions) #number of observable states
        self.make_states_dict()

    def make_states_dict(self): #value->index mapping for hidden states and emission matrices only since their values are unique
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq): #seq -> observation seen
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        #num columns = #of observations made
        #rows = #of hidden states
        seq_len = len(seq)
        dp = [[float("-inf") for j in range(seq_len)] for i in range(self.N)]
        print("dp shape",np.matrix(dp).shape)
        print("B shape ",np.matrix(self.B).shape)
        print("A shape ",np.matrix(self.A).shape)

        #initialize
        init_obs_state_name = seq[0]
        init_obs_state_indx = self.emissions_dict[init_obs_state_name]
        dp[0][0] = self.pi[0]*self.B[0][init_obs_state_indx]
        dp[1][0] = self.pi[1]*self.B[1][init_obs_state_indx]
        
        #pointer to hold the state recently entered into the queue
        path_backtrack = deque()
        seq_pointer = 0

        #dp iteration
        for j in range(1,seq_len): #j->mood we are in
            dest_row=0
            for i in range(0,self.N): #i->hidden state we are in
                print("indices",(i,j))
                mood_name=seq[j]
                mood_indx = self.emissions_dict[mood_name]
                
                print("incoming prob",dp[i][j-1],"*",self.A[i][dest_row],"*",self.B[dest_row][mood_indx],"=",self.B[dest_row][mood_indx]*self.A[i][dest_row]*dp[i][j-1])
                input()

                incoming_probability = self.B[dest_row][mood_indx]*self.A[i][dest_row]*dp[i][j-1] #this can be optimized by multiplying sef.B[i][mood_indx] in the end
                dp[i][j] = max(dp[i][j],incoming_probability)
                print("dp",dp)
                #reset dest_row
            print("dp",dp)
            dest_row=(dest_row+1)%2


if __name__=="__main__":
    def test_1():
        '''
        Bob's observed mood (Happy or Grumpy)  - observed states
        can be modelled with the weather (Sunny, Rainy) - hidden states
        '''

        '''
        transmission probabilities : rows->source and column->dest
        '''
        A = np.array([
            [0.8, 0.2],
            [0.4, 0.6]
        ])

        HS = ['Sunny', 'Rainy']
        O = ["Happy", 'Grumpy']

        priors = [2/3, 1/3] # starting probabilties

        '''
        emission probabilities of different observable states
        '''
        B = np.array([
            [0.8, 0.2],
            [0.4, 0.6]
        ])

        ES = ["Happy", "Grumpy", "Happy"] # observation  seeen
        
        model = HMM(A, HS, O, priors, B)
        seq = model.viterbi_algorithm(ES)
        assert (seq == ['Sunny', 'Sunny', 'Sunny'])

    test_1()