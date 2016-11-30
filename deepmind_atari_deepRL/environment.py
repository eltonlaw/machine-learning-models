import sys
sys.path.append("./gym")
import gym
atari = ["SpaceInvaders-v0"]
env = gym.make(atari[0])

actions = env.action_space
#observation_space = env.observation_space

for i_episode in range(20):
    observation = env.reset() # First observation resets
    for t in range(100):
        env.render() # Render each step
        print observation
        action = actions.sample()
        observation,reward,done,info = env.step(action)
        if done:
            print "episode done after %d timesteps" % t
            break
