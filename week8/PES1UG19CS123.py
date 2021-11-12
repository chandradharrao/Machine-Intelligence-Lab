import numpy as np

'''
######CHECK############
optimization : using numpy array
multiply in the end
check for underflow
deque pop time complexity?
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

        #initialize
        init_obs_state_name = seq[0]
        init_obs_state_indx = self.emissions_dict[init_obs_state_name]
        for i in range(0,self.N): #iterate through all the states for seq[0]
            dp[i][0] = self.pi[i]*self.B[i][init_obs_state_indx]
        
        #store states recently entered into the queue
        path_backtrack = []

        #dp iteration
        for j in range(1,seq_len): #j->mood we are in
            column_max_incoming_prob = float("-inf") #for each column store the max prob of incoming state
            column_max_state=None
            for i in range(0,self.N): #i->hidden state we are in
                mood_name=seq[j]
                mood_indx = self.emissions_dict[mood_name]

                for k in range(0,self.N): #k->from all the N prev states
                    incoming_probability = self.B[i][mood_indx]*self.A[k][i]*dp[k][j-1] #this can be optimized by multiplying sef.B[i][mood_indx] in the end

                    if incoming_probability>dp[i][j]: 
                        dp[i][j] = incoming_probability
                        if column_max_incoming_prob<dp[i][j]:
                            column_max_state=self.states[k]
                            column_max_incoming_prob=dp[i][j]
            path_backtrack.append(column_max_state) #update at the end of each column calculation

        #choose last column max prbability state
        max_prob = float("-inf")
        to_insert = None
        for i in range(0,self.N):
            if dp[i][-1]>max_prob:
                max_prob=dp[i][-1]
                to_insert = self.states[i]
        path_backtrack.append(to_insert)

        return path_backtrack
