import numpy as np
from collections import defaultdict
from scipy.special import expit



class TermFunc:
    def __init__(self, nb_states, lr):
        self.nb_states = nb_states
        self.lr = lr
        self.table = defaultdict()


    # Caution the check for termination needs to be done with sp not s
    def check_term(self, s, bool=False):
        if s not in self.table:
            self.table[s] = 0
        if bool:
            return self.table[s]
        return (np.random.random() < expit(self.table[s]))
    

    def get_term_val(self, s):
        return expit(self.check_term(s,True))
        

    # Update formula needs to be controlled again
    def update(self, s, A):
        self.check_term(s,True)
        self.table[s] -= self.lr*(expit(self.table[s])*(1-expit(self.table[s])))*A

    