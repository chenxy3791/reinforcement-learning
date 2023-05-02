# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:46:15 2023

@author: chenxy
"""
import numpy as np

states  = [      (0,1),(0,2),(0,3), 
           (1,0),(1,1),(1,2),(1,3), 
           (2,0),(2,1),(2,2),(2,3), 
           (3,0),(3,1),(3,2)        ]                 
s_term  = tuple([(0,0),(3,3)]) # Terminal states

actions = ['UP','DOWN','LEFT','RIGHT']
def policy(s,a):
    '''
    The agent follows the equiprobable random policy (all actions equally likely).
    '''
    prob_s_a = 1.0 / len(actions)
    
    return prob_s_a

def transit_to(s,a):
    '''
    next_state to which when taking action "a" from the current state "s"
    '''
    s_next = list(s)
    if a == 'UP':
        s_next[0] = s[0] - 1
    elif a == "DOWN":
        s_next[0] = s[0] + 1
    elif a == "LEFT":
        s_next[1] = s[1] - 1
    else:
        s_next[1] = s[1] + 1
    
    if s_next[0] < 0 or s_next[0] > 3 or s_next[1] < 0 or s_next[1] > 3:
        s_next = list(s)
    
    return tuple(s_next)

def reward_func(s):
    '''
    The immediate reward when transit from other states to this state.
    Reward is -1 for all transitions.
    '''
    # return 0 if s in s_term else -1
    return -1

# # Test transit_to()
# s = (1,2)
# for a in actions:    
#     print('transit_to({0},{1}) = {2}'.format(s,a,transit_to(s, a)))

# # Test policy()
# for i in range(states.shape[0]):
#     for j in range(states.shape[1]):     
#         s = (i,j)
#         for a in actions:    
#             print('policy({0},{1} = {2})'.format(s,a,policy(s,a)))
            
# # Test reward_func()            
# for i in range(states.shape[0]):
#     for j in range(states.shape[1]):     
#         s = (i,j)
#         print('reward_func({0} = {1})'.format(s, reward_func(s)))

# Initialize v(s) to all-zeros
v = np.zeros((4,4))

iter_cnt = 0    
while True:
    max_delta = 0    
    for (i,j) in states:
        s = (i,j)
        new_v = 0
        for a in actions:
            p_action = policy(s,a)
            s_next   = transit_to(s,a)
            reward   = reward_func(s_next)
            new_v    = new_v + p_action * (reward + v[s_next[0],s_next[1]])
            
        max_delta= max(max_delta, abs(new_v - v[i][j]))
        v[i][j] = new_v
    iter_cnt = iter_cnt + 1
    # print('iter_cnt = {0}, max_delta = {1}, v = {2}'.format(iter_cnt, max_delta, v))
    if max_delta < 0.001 or iter_cnt == 100:        
        break

# How to specify format when printing np numeric array?
print('iter_cnt = {0},  v = \n {1}'.format(iter_cnt, v))


    
    