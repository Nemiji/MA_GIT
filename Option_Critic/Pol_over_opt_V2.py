import numpy as np
from collections import defaultdict



class Higherlvl:
    def __init__(self, nb_states, nb_opt, lr, eps, gamma):
        self.nb_states = nb_states
        self.nb_opt = nb_opt
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.table = defaultdict()

    def choose_opt(self, state, opt=None):
        if state not in self.table:
            self.table[state] = np.zeros(self.nb_opt)
        if opt != None:
            return self.table[state][opt]
        if (np.random.random() < self.eps):
            return np.random.randint(0, self.nb_opt)
        else:
            return np.argmax(self.table[state])

    def update_table(self, r, s, sp, opt):
        self.choose_opt(sp,opt)
        self.table[s][opt] += self.lr*(r+self.gamma*np.max(self.table[sp])-self.choose_opt[s][opt])