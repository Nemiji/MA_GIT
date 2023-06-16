import gym
import numpy as np
from Option_Critic.Pol_over_opt_V2 import Higherlvl
from Option_Critic.SigmoidTerm_V2 import TermFunc
from Option_Critic.Critic_V3 import Critic
from Option_Critic.IntraPol_V4_LOC import IntraPol_LOC
from Option_Critic.IntraPol_V4_REM import IntraPol_REM
from Option_Critic.IntraPol_V4_CONN import IntraPol_CONN



import sys
import logging
import cyberbattle._env.cyberbattle_env
import cyberbattle.agents.baseline.agent_wrapper as w

#-----------------------------------------------------

env = gym.make('CyberBattleToyCtf-v0')


# Hyperparams
ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count = 12,
    maximum_total_credentials=12,
    identifiers=env.environment.identifiers
)




nb_iters = 2500
nb_steps = 1000

# Gives a multidiscrete([7000]). What are the right features for both spaces ?
nb_states = w.HashEncoding(ep,[w.Feature_active_node_properties(ep),w.Feature_active_node_age(ep),7000])
nb_actions_max = 0

nb_opt = 3
gamma = 0.99
epsilon = 1e-1
lr_term = 0.25
lr_intra = 0.25
lr_critic = 0.5
lr_higher = 0.3
temp = 1e-2

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


higher_lvl = Higherlvl(nb_states,nb_opt,lr_higher,epsilon,gamma)
term_list = [TermFunc(nb_states, lr_term) for _ in range(nb_opt)]
intra_list = []
for i in range(nb_opt):
    intra_list.append(IntraPol_LOC(nb_states,lr_intra, temp, ep))
    intra_list.append(IntraPol_REM(nb_states, lr_intra, temp, ep))
    intra_list.append(IntraPol_CONN(nb_states, lr_intra, temp, ep))
nb_actions_max = len(intra_list[0].loc_table) + len(intra_list[1].rem_table) + len(intra_list[2].connection_table)
critic = Critic(nb_states,nb_actions_max,nb_opt,lr_critic,gamma)

#-----------------------------------------------------------------------------

R_list = []
steps_list = []

for it in range (nb_iters):
    S = 0
    R = 0
    done = False
    s = env.reset()

    opt = higher_lvl.choose_opt(s)

    for step in range(nb_steps):
        a = intra_list[opt].choose_action(s)
        sp,r,done,info = env.step(a)
        R += r
