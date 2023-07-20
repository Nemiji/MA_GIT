import numpy as np
import gym
from gym import spaces
from gym.spaces import Box, Discrete
from collections import defaultdict
import hashlib
import random


class PenTestSim(gym.Env):

    def __init__(self):

        self.small_1 = [[[0,1,0,0,0,0],[1,0,1,1,0,0],[0,1,0,1,1,0],[0,1,1,0,0,1],[0,0,1,0,0,0],[0,0,0,1,0,0]],4,[2]]
        self.small_2 = [[[0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0],[1,1,1,0,1,1,0,0],[0,0,0,1,0,1,0,0],[0,0,0,1,1,0,1,1],[0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0]],6,[3,5]]
        self.enterprise_2 = [[[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]],14,[3,6,10,11,13,16,21]]
        self.nb_nodes = len(self.small_1[0])
        # Three action types: Discovery (==0) or Connection (==1) or Priv_Esc (==2), Targeted Node
        self.action_space = spaces.MultiDiscrete([3,self.nb_nodes])
        self.observation_space = spaces.Dict({
            'available_nodes' : spaces.MultiBinary(self.nb_nodes),
            'current_node' : spaces.Discrete(self.nb_nodes),
            'node_scan_status' : spaces.Discrete(2),
            'escalation_status' : spaces.Discrete(2),
            'importance_lvl' : spaces.Discrete(2)
        })
        self.goal_id = self.small_1[1]
        self.current_node = 0
        self.important_nodes = self.small_1[2]
        self.node_conns = [[0,1,0,0,0,0],[1,0,1,1,0,0],[0,1,0,1,1,0],[0,1,1,0,0,1],[0,0,1,0,0,0],[0,0,0,1,0,0]]
        self.node_list = self._init_node_list()

    def reset(self):
        self.node_list = self._init_node_list()
        self.current_node = 0
        for i in range(self.nb_nodes):
            self.node_list[i]['discovered'] = 0
            self.node_list[i]['escalation_status'] = 0
        obs = {
            'available_nodes' : np.zeros(self.nb_nodes,dtype=int).tolist(),
            'current_node' : self.current_node,
            'node_scan_status' : self.node_list[self.current_node]['discovered'],
            'escalation_status' : self.node_list[self.current_node]['escalation_status'],
            'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
        }
        return obs


    def step(self,action):
        r = 0
        done = False
        if action[0] == 0:
            if action[1] != self.current_node:
                obs = {
                    'available_nodes' : np.zeros(self.nb_nodes,dtype=int).tolist(),
                    'current_node' : self.current_node,
                    'node_scan_status' : 0,
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
                return obs, r, done, {"Info" : "Cannot scan node without being in it"}
            else:
                self.node_list[self.current_node]['discovered'] = 1
                obs = {
                    'available_nodes' : self.node_conns[self.current_node],
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
                return obs, r, done, {"Info" : "Node scanned succesfully"}
        
        if action[0] == 1:
            if self.node_list[self.current_node]['discovered'] != 1 :
                obs = {
                    'available_nodes' : np.zeros(self.nb_nodes,dtype=int).tolist(),
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
                return obs, r, done, {"Info" : "Cannot connect to undiscovered node"}
            
            if self.node_list[self.current_node]['importance_lvl'] == 1:
                if self.node_list[self.current_node]['escalation_status'] == 0:
                    obs = {
                    'available_nodes' : self.node_conns[self.current_node],
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
                    return obs, r, done, {"Info" : "Connection succesfull"}
                if self.node_conns[self.current_node][action[1]] == 1:
                    self.current_node = action[1]
                    if self.current_node == self.goal_id:
                        r = 1
                        done = True
                    obs = {
                        'available_nodes' : self.node_conns[self.current_node],
                        'current_node' : self.current_node,
                        'node_scan_status' : self.node_list[self.current_node]['discovered'],
                        'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                        'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                    }
                    return obs, r, done, {"Info" : "Connection succesfull"}
                if self.node_conns[self.current_node][action[1]] != 1:
                    obs = {
                        'available_nodes' : self.node_conns[self.current_node],
                        'current_node' : self.current_node,
                        'node_scan_status' : self.node_list[self.current_node]['discovered'],
                        'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                        'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                    }
                    return obs, r, done, {"Info" : "Connection impossible"}
            if self.node_list[self.current_node]['importance_lvl'] != 1:
                if self.node_conns[self.current_node][action[1]] == 1:
                    self.current_node = action[1]
                    if self.current_node == self.goal_id:
                        r = 1
                        done = True
                    obs = {
                        'available_nodes' : self.node_conns[self.current_node],
                        'current_node' : self.current_node,
                        'node_scan_status' : self.node_list[self.current_node]['discovered'],
                        'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                        'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                    }
                    return obs, r, done, {"Info" : "Connection succesfull"}
                if self.node_conns[self.current_node][action[1]] != 1:
                    obs = {
                        'available_nodes' : self.node_conns[self.current_node],
                        'current_node' : self.current_node,
                        'node_scan_status' : self.node_list[self.current_node]['discovered'],
                        'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                        'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                    }
                    return obs, r, done, {"Info" : "Connection impossible"}
                    
        
        if action[0] == 2:
            if action[1] != self.current_node:
                obs = {
                    'available_nodes' : np.zeros(self.nb_nodes,dtype=int).tolist(),
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
                return obs, r, done, {"Info" : "Cannot do privilege escaltion in node without being present in it"}

            self.node_list[self.current_node]['escalation_status'] = 1
            if self.node_list[self.current_node]['discovered'] != 1:
                obs = {
                    'available_nodes' : np.zeros(self.nb_nodes,dtype=int).tolist(),
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
            else:
                obs = {
                    'available_nodes' : self.node_conns[self.current_node],
                    'current_node' : self.current_node,
                    'node_scan_status' : self.node_list[self.current_node]['discovered'],
                    'escalation_status' : self.node_list[self.current_node]['escalation_status'],
                    'importance_lvl' : self.node_list[self.current_node]['importance_lvl']
                }
            return obs, r, done, {"Info" : "Privilege escalation succesfull"}
    
    def valid_action_sample(self):
        valid_actions = []
        if self.node_list[self.current_node]['discovered'] == 1:
            for i in range(self.nb_nodes):
                if self.node_conns[self.current_node][i] == 1:
                    valid_actions.append(i)
            return (1,np.random.choice(valid_actions))
        return (0,self.current_node)
    
        
    def _init_node_list(self):
        node_list = defaultdict()
        for i in range(self.nb_nodes):
            node_list[i] = {'discovered' : 0, 'connections' : self.node_conns[i], 'escalation_status' : 0, 'importance_lvl' : 0}
            for j in range(len(self.important_nodes)):
                if i == self.important_nodes[j]:
                    node_list[i]['importance_lvl'] = 1
        return node_list
    
    def hashed_obs(self, obs):
        stringed_obs = str(obs)
        return hashlib.md5(stringed_obs.encode()).hexdigest()
    
    def change_network(self, env):
        if env == 'small_1':
            self.nb_nodes = len(self.small_1[0])
            self.node_conns = self.small_1[0]
            self.goal_id = self.small_1[1]
            self.important_nodes = self.small_1[2]
            self.observation_space = spaces.Dict({
            'available_nodes' : spaces.MultiBinary(self.nb_nodes),
            'current_node' : spaces.Discrete(self.nb_nodes),
            'node_scan_status' : spaces.Discrete(2),
            'escalation_status' : spaces.Discrete(2),
            'importance_lvl' : spaces.Discrete(2)
            })
            self.action_space = spaces.MultiDiscrete([3,self.nb_nodes])
            print("Changed to ENV: SMALL_1")
        elif env == 'small_2':
            self.nb_nodes = len(self.small_2[0])
            self.node_conns = self.small_2[0]
            self.goal_id = self.small_2[1]
            self.important_nodes = self.small_2[2]
            self.observation_space = spaces.Dict({
            'available_nodes' : spaces.MultiBinary(self.nb_nodes),
            'current_node' : spaces.Discrete(self.nb_nodes),
            'node_scan_status' : spaces.Discrete(2),
            'escalation_status' : spaces.Discrete(2),
            'importance_lvl' : spaces.Discrete(2)
            })
            self.action_space = spaces.MultiDiscrete([3,self.nb_nodes])
            print("Changed to ENV: SMALL_2")
        elif env == 'enterprise_2':
            self.nb_nodes = len(self.enterprise_2[0])
            self.node_conns = self.enterprise_2[0]
            self.goal_id = self.enterprise_2[1]
            self.important_nodes = self.enterprise_2[2]
            self.observation_space = spaces.Dict({
            'available_nodes' : spaces.MultiBinary(self.nb_nodes),
            'current_node' : spaces.Discrete(self.nb_nodes),
            'node_scan_status' : spaces.Discrete(2),
            'escalation_status' : spaces.Discrete(2),
            'importance_lvl' : spaces.Discrete(2)
            })
            self.action_space = spaces.MultiDiscrete([3,self.nb_nodes])
            print("Changed to ENV: ENTERPRISE_2")

        