# Solution for Udacity ML Nanodegree P4
def update(self, t):
		# Gather inputs
		self.alpha = 0.80**(t+2) # Learning Rate, should decrease over time as you get a bigger and bigger knowledge base
		self.gamma = 0.80**(t+2) # 0.95**(t+4)# Discount Factor, should decrease over time as you value short term reward
		self.epsilon = 0.80**(t+2) # Probability of exploration, should decrease over time as we get more confident in our q_values
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self) # Passes yourself in as the agent returns a dictionary of inputs: [Lights,Oncoming,Left,Right]
		deadline = self.env.get_deadline(self)

		if 0 < deadline < 15:
			deadline_class = "urgent"
		else:
			deadline_class = "not urgent"

		self.state = (self.next_waypoint,inputs["light"],inputs["oncoming"],inputs["left"],inputs["right"],deadline_class)

		if random.random() > self.epsilon: #Probability of 1-self.epsilon of Exploitation
			action = max(self.q_table[self.state], key=self.q_table[self.state].get) # The key(one of the actions) taken from the maximum q_value value(reward) is the action we will take when exploiting
		else: # Exploration - If random number less than probability of exploration. Select a random action
			random_int = int(math.floor(random.random()*len(self.valid_actions))) #random_int = 0,1,2 or 3
			action = self.valid_actions[random_int]

		# Execute action and get reward. Returns scalar value
		reward = self.env.act(self, action)

		# Get next state
		new_deadline = self.env.get_deadline(self)
		if 0 < new_deadline < 15:
			new_deadline_class = "urgent"
		else:
			new_deadline_class = "not urgent"

		next_next_waypoint = self.planner.next_waypoint()
		next_inputs = self.env.sense(self)
		next_state = (next_next_waypoint,next_inputs["light"],next_inputs["oncoming"],next_inputs["left"],next_inputs["right"],new_deadline_class) # Next State tuple
		next_state_actions = self.q_table[next_state] # State you land in from taking the action, should return a dictionary with 4 key-values(the 4 available actions you can take in this next state)
		optimal_future_reward = max(next_state_actions.values()) #Finds the max q value for the next state

		self.q_table[self.state][action] = (1-self.alpha)*self.q_table[self.state][action] + self.alpha * (reward + self.gamma * optimal_future_reward)
