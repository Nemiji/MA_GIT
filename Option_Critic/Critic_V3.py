import numpy as np
from collections import defaultdict


class Critic:
    def __init__(self, nb_states, nb_actions, nb_opt, lr, gamma, higher_lvl):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_opt = nb_opt
        self.lr = lr
        self.gamma = gamma
        self.higher_lvl = higher_lvl
        self.Q_U = defaultdict()
    

    def get_Q_Omega_val(self, state, opt=None):
        if state not in self.higher_lvl:
            self.higher_lvl[state] = np.zeros(self.nb_opt)
        if opt != None:
            return self.higher_lvl[state][opt]
        return self.higher_lvl[state]


    def get_Q_U_val(self, s, opt, a):
        if s not in self.Q_U:
            self.Q_U[s] = np.zeros((self.nb_opt, self.nb_actions))
        return self.Q_U[s][opt,a]


    # Advantage(sp, opt) = Q_Omega(sp, opt) - V_omega(sp)

    def advantage(self, s, opt):
        return self.get_Q_Omega_val(s,opt) - np.max(self.higher_lvl[s])


    #Updates Q_Omega (A_Omega) and Q_U following TD update
    def TD_update(self, r, s, sp, a, opt, done, term):
        self.get_Q_U_val(s,opt,a) # To make sure the state is in Q_U
        self.get_Q_Omega_val(s,opt) # To make sure state is in higher_lvl
        delta = r #- self.get_Q_U_val(s,opt,a)
        if not done:
            delta += self.gamma*(1-term.get_term_val(sp))*self.get_Q_Omega_val(sp,opt) + self.gamma*term.get_term_val(sp)*np.max(self.get_Q_Omega_val(sp))
        self.Q_U[s][opt,a] += self.lr * (delta - self.get_Q_U_val(s,opt,a))
        self.higher_lvl[s][opt] += self.lr * (delta - self.get_Q_Omega_val(s,opt))
