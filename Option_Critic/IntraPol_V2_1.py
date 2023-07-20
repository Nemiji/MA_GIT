import numpy as np
from collections import defaultdict
from scipy.special import expit, logsumexp, softmax


class IntraPol:
    def __init__(self, nb_states, nb_actions, lr, temp):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.lr = lr
        self.table = defaultdict()
        self.temp = temp


    def choose_action(self, s, a=None):
        if s not in self.table:
            self.table[s] = np.zeros(self.nb_actions)
        if a!= None:
            return self.table[s][a]
        return np.random.choice(self.nb_actions, p=softmax(self.table[s]/self.temp))
    
    def get_intra_val(self, s, a):
        return softmax(self.choose_action(s,a)/self.temp)
    
    #Does more or less the same thing as the paper but needs reworking
    def update(self, s, a, Q_U):
        tmp = softmax(self.table[s]/self.temp)
        self.table[s] -= self.lr*tmp*Q_U
        self.table[s][a] += self.lr*Q_U