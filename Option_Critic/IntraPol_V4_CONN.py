import numpy as np
from collections import defaultdict
from scipy.special import expit, logsumexp, softmax


class IntraPol_CONN:
    def __init__(self, nb_states, lr, temp, ep):
        self.nb_states = nb_states
        self.lr = lr
        self.table = defaultdict()
        self.max_nodes = ep.maximum_node_count
        self.max_ports = ep.port_count
        self.max_creds = ep.maximum_total_credentials
        self.connection_table = self.init_connection_table()
        self.temp = temp
  
    
    def init_connection_table(self):
        self.connection_table = []
        for i in range(self.max_nodes):
            for j in range(self.max_nodes):
                for k in range(self.max_ports):
                    for l in range(self.max_creds):
                        # Source and Dest must be different
                        if i != j:
                            self.connection_table.append((i,j,k,l))



    def choose_action(self, s, a=None):
        if s not in self.table:
            self.table[s] = np.zeros(len(self.connection_table))
        if a!= None:
            return self.table[s][a]
        return np.random.choice(len(self.connection_table), p=softmax(self.table[s]/self.temp))
    
    
    def get_intra_val(self, s, a):
        return softmax(self.choose_action(s,a)/self.temp)
    
    #Does more or less the same thing as the paper but needs reworking
    def update(self, s, a, Q_U):
        tmp = softmax(self.table[s]/self.temp)
        self.table[s] -= self.lr*tmp*Q_U
        self.table[s][a] += self.lr*Q_U