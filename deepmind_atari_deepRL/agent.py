class Agent:
    def __init__(self,valid_states,valid_actions):
        self.valid_states = valid_states
        self.valid_actions = valid_actions
        self.Q = np.zeros(len(valid_states),len(valid_actions))
    def get_action(self,state,t):
        self.epsilon = 1/t # Probability of Exploration
        # Exploitation
        if np.random.rand() > self.epsilon:
            action = max(Q[current_state])
        # Exploration
        else: 
            ii = np.randomint(0,high=len(self.valid_actions))
            action =  self.valid_actions[ii]
        return action
    def update(self,state,action,t):
        ## All three values should slowly decrease after each episode
        self.alpha = 1/t # Learning Rate
        self.gamma= 1/t # Discount
        
        ## Take Action
        ## Note new state
        ## Get Reward and update Q_table


