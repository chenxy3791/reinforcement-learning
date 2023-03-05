# Author: Chenxy
# Modified based on https://github.com/icoxfog417/baby-steps-of-rl-ja/DP/{environment.py,bellman_equation.py}.
# 2023-03-05: First creation

from enum import Enum
import numpy as np
import random

class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    # Opposite numeric values are assigned to opposite direction.
    # This is for the convenience of implementation in transit_func().
    UP    = 1
    DOWN  = -1
    LEFT  = 2
    RIGHT = -2

class Environment():

    def __init__(self, grid, move_prob=1.0):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        # grid[i][j] = 1: Terminate cell (game end)
        # grid[i][j] =-1: Cliff cells
        # grid[i][j] = 0: Other cells

        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -1

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        # move_prob = 1.0 means agent always move in the selected direction.
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        '''
        valid states.
        In CliffWalking-v0 games, cliff cells are not valid cells (unreachable).
        '''
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Cliff cells are not included to the valid state list.
                if not(row == (self.row_length - 1) and 0 < column < (self.column_length - 1)):
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        """
        Prob(s',r|s,a) stored in one dict[(s',reward)].
        """
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state.row == (self.row_length - 1) and 0 < next_state.column < (self.column_length - 1):
                reward = -100
                next_state = State(self.row_length - 1, 0) # Return to start grid when falls into cliff grid.
            else:
                reward = -1
            
            if (next_state,reward) not in transition_probs:
                transition_probs[(next_state,reward)] = prob
            else:
                transition_probs[(next_state,reward)] += prob

        return transition_probs

    def can_action_at(self, state):
        '''
        Assuming:
            grid[i][j] = 1: Terminate grid
            grid[i][j] =-1: Cliff grids
            grid[i][j] = 0: Other grids
        '''
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        """
        Predict the next state upon the combination of {state, action}
        {state, action} --> next_state
        Called in transit_func()
        """
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Entering into cliff grids is related to the correspong penalty and 
        # reset to start grid, hence will be handled upper layer.

        return next_state

    def reset(self):
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    # def step(self, action):
    #     next_state, reward, done = self.transit(self.agent_state, action)
    #     if next_state is not None:
    #         self.agent_state = next_state
    # 
    #     return next_state, reward, done
    # 
    # def transit(self, state, action):
    #     transition_probs = self.transit_func(state, action)
    #     if len(transition_probs) == 0:
    #         return None, None, True
    # 
    #     next_states = []
    #     probs = []
    #     for (s,reward) in transition_probs:
    #         next_states.append((s,reward))
    #         probs.append(transition_probs[(s,reward)])
    # 
    #     (next_state,reward) = np.random.choice(next_states, p=probs)
    #     done = (next_state.row == self.row_length - 1) and (next_state.column == self.column_length - 1)
    #     return next_state, reward, done

class Planner():

    def __init__(self, env):
        self.env     = env
        self.log     = []
        self.V_grid  = []
        self.iters   = 0

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        '''
        Maybe moved to Environment in the future.
        '''
        transition_probs = self.env.transit_func(state, action)
        for (next_state,reward) in transition_probs:
            prob = transition_probs[(next_state,reward)]
            # reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        """
        Convert dict to 2-D array specific to grid-world-like game, for the convenience of 
        print_value_grid(), etc.
        Using numpy array maybe better.
        """
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]
    
        return grid
    
    def print_value_grid(self):
        for i in range(len(self.V_grid)):
            for j in range(len(self.V_grid[0])):
                print('{0:6.3f}'.format(self.V_grid[i][j]), end=' ' )
            print('')

# class ValueIterationPlanner(Planner):
# 
#     def __init__(self, env):
#         super().__init__(env)
# 
#     def plan(self, gamma=0.9, threshold=0.0001):
#         self.initialize()
#         actions = self.env.actions
#         V = {}
#         for s in self.env.states:
#             # Initialize each state's expected reward.
#             V[s] = 0
# 
#         while True:
#             delta = 0
#             self.log.append(self.dict_to_grid(V))
#             for s in V:
#                 if not self.env.can_action_at(s):
#                     continue
#                 expected_rewards = []
#                 for a in actions:
#                     r = 0
#                     for prob, next_state, reward in self.transitions_at(s, a):
#                         r += prob * (reward + gamma * V[next_state])
#                     expected_rewards.append(r)
#                 max_reward = max(expected_rewards)
#                 delta = max(delta, abs(max_reward - V[s]))
#                 V[s] = max_reward
# 
#             self.V_grid = self.dict_to_grid(V)            
#             self.iters = self.iters + 1
#             print('ValueIteration: iters = {0}'.format(self.iters))
#             self.print_value_grid()
#             print('******************************')
#             
#             if delta < threshold:
#                 break

class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # Initialize policy.
                # At first, each action is taken uniformly. 
                # Any other random initialization should be also OK, for example, gaussian distribution
                self.policy[s][a] = 1 / len(actions)                                    

    def policy_evaluation(self, gamma, threshold):
        V = {}
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += action_prob * prob * \
                             (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        Implement the policy iteration algorithm
        gamma    : discount factor
        threshold: delta for policy evaluation convergency judge.
        """
        self.initialize()
        states  = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            # Estimate expected rewards under current policy.
            V = self.policy_evaluation(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # Get an action following to the current policy.
                policy_action = take_max_action(self.policy[s])

                # Compare with other actions.
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r
                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # Update policy (set best_action prob=1, otherwise=0 (greedy))
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            # Turn dictionary to grid
            self.V_grid = self.dict_to_grid(V)
            self.iters = self.iters + 1
            print('PolicyIteration: iters = {0}'.format(self.iters))
            self.print_value_grid()
            print('******************************')

            if update_stable:
                # If policy isn't updated, stop iteration
                break

    def print_policy(self):
        print('PolicyIteration: policy = ')
        actions = self.env.actions
        states  = self.env.states
        for s in states:
            print('\tstate = {}'.format(s))
            for a in actions:
                print('\t\taction = {0}, prob = {1}'.format(a,self.policy[s][a]))

        # Optimal actions
        action_array = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            action_array.append(row)
        for s in states:
            max_prob = -1                  
            for a in actions:
                if self.policy[s][a] > max_prob:
                    max_prob = self.policy[s][a]
                    opt_action = a
            action_array[s.row][s.column] = opt_action.value
        
        print('PolicyIteration: optimal policy = ')
        for i in range(self.env.row_length):
            print("========================")
            for j in range(self.env.column_length):
                if action_array[i][j] == Action.UP.value:
                    print('  UP   ', end='')
                elif action_array[i][j] == Action.DOWN.value:
                    print(' DOWN  ', end='')
                elif action_array[i][j] == Action.LEFT.value:
                    print(' LEFT  ', end='')
                elif action_array[i][j] == Action.RIGHT.value:
                    print(' RIGHT ', end='')
                else:
                    print('   X   ', end='')
            print('')
                
if __name__ == "__main__":

    # Create grid environment
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    grid[3][11] = 1 # Terminate cell
    for k in range(1,11):
        grid[3][11] = -1  # Cliff cells
        
    # # A smaller grid environment, only for the convenience of debug.
    # grid = [
    #     [0,  0, 0],
    #     [0,  0, 0],
    #     [0, -1, 1]
    # ]    
    
    env2 = Environment(grid)
    policyIterPlanner = PolicyIterationPlanner(env2)
    policyIterPlanner.plan(0.9,0.001)
    policyIterPlanner.print_value_grid()    
    policyIterPlanner.print_policy()    