import environment
import numpy as np

q = np.empty((n_states,n_actions))
valid_actions = env.get_valid_actions()

for t in range(1000):
    """ 
    State_t -> Action_t -> (Reward_t , State_t+1) ->...
    """
    alpha = 1/t # Learning Rate
    gamma = 1/t # Discount
    epsilon = 1/t # Probability of Exploration

    state = env.get_current_state()

    if np.random.rand() > epsilon:
        # Exploitation
        action = max(q[current_state])
    else: 
        # Exploration
        ii = np.randomint(0,high=len(valid_actions))
        action =  random_valid_action[ii]



