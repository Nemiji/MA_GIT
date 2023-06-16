import numpy as np
from collections import defaultdict
from scipy.special import expit, logsumexp, softmax


class IntraPol_LOC:
    def __init__(self, nb_states, lr, temp,  ep):
        self.nb_states = nb_states
        self.lr = lr
        self.table = defaultdict()
        self.max_nodes = ep.maximum_node_count
        self.max_local = ep.local_attacks_count
        self.loc_table = self.init_loc_table()
        self.temp = temp



    def init_loc_table(self):
        self.loc_table = []
        for i in range(self.max_nodes):
            for j in range(self.max_local):
                self.loc_table.append((i,j))



    def choose_action(self, s, a=None):
        if s not in self.table:
            self.table[s] = np.zeros(len(self.loc_table))
        if a!= None:
            return self.table[s][a]
        a = np.random.choice(len(self.loc_table), p=softmax(self.table[s]/self.temp))
        return self.loc_table[a]
    
    
    def get_intra_val(self, s, a):
        return softmax(self.choose_action(s,a)/self.temp)
    
    #Does more or less the same thing as the paper but needs reworking
    def update(self, s, a, Q_U):
        tmp = softmax(self.table[s]/self.temp)
        self.table[s] -= self.lr*tmp*Q_U
        self.table[s][a] += self.lr*Q_U