# Modified based on https://github.com/icoxfog417/baby-steps-of-rl-ja/DP/environment_demo.py.

import random
from maze_env import Environment, Agent, State
import time
from planner import ValueIterationPlanner

def play(grid, num_episodes):
    env = Environment(grid)
    agent = Agent(env,5)

    # Try 10 games (one game is one episode).
    for i in range(num_episodes):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False
    
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
    
        print("Episode {}: Agent gets {:6.2f} reward.".format(i, total_reward))

def value_evaluation_all_states(grid, max_steps=7):
    for k in range(max_steps):    
        print('================================================')        
        print('max_steps = {0}'.format(k))        
        env = Environment(grid)
        agent = Agent(env,k)
        
        t_start = time.time()
        for i in range(len(grid)):
            for j in range(len(grid[0])):            
                s = State(i,j)
                print('s = {0}, agent.V(s) = {1:6.3f}'.format(s, agent.V(s,0)))
        t_stop = time.time()
        print('time cost = {0:6.2f}(sec)'.format((t_stop-t_start)))
        print('')

def value_evaluation_one_state(grid, s):
    for max_steps in range(8):    
        print('================================================')        
        print('max_steps = {0}'.format(max_steps))        
        env = Environment(grid)
        agent = Agent(env,max_steps)
        
        t_start = time.time()
        print('s = {0}, agent.V(s) = {1:6.3f}'.format(s, agent.V(s,0)))
        t_stop = time.time()
        print('time cost = {0:6.2f}(sec)'.format((t_stop-t_start)))
        print('')

if __name__ == "__main__":

    # Create grid environment
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    
    # A smaller grid environment, only for the convenience of debug.
    # grid = [
    #     [9, 0, 1],
    #     [0, 0, -1]
    # ]    
    
    # play(grid, 10)
    # 
    # value_evaluation_all_states(grid, 7)
    # 
    # s = State(len(grid)-1,0) # Start from left-bottom cell
    # value_evaluation_one_state(grid, s)
    
    env = Environment(grid)
    valueIterPlanner = ValueIterationPlanner(env)
    valueIterPlanner.plan(0.9,0.001)
    valueIterPlanner.print_value_grid()
    

        
        