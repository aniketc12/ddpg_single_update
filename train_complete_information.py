from env.pid_comlete_information_env import PidEnvSingle
import torch
import numpy as np
from ddpg import Agent
from OUNoise import Noise
import matplotlib.pyplot as plt

batch_size = 128
rewards = []
avg_rewards = []
env = PidEnvSingle()
agent = Agent(num_states=3, num_actions=3, gamma=0.99)
noise = Noise(num_actions=3)
zeros = [0]
normalized = []

for episode in range(1):
    noise.reset()
    eps_reward = 0
    setpoint = 20
    exit = False
    best_parameters = [(0,0,0), 0, 0]
    step = 0
    while step <= 1000:
        step += 1
        print('Episode: '+str(episode)+' Step: '+str(step))
        state = env.reset(setpoint)
        action_before = agent.get_action(state)
        action = noise.get_action(action_before, step)

        # Environment completes whoel episode internally but instead of just returning
        # the final state and final reward, it includes all encountered states and rewards as two lists
        new_states, new_rewards = env.step(action)
        agent.mem.push(state, action, new_rewards[0], new_states[0])

        # Step through the two lists that the environment returned and add the data to the Replay Buffer
        #for ind in range(len(new_states)-1):
        #    #If reward is too small then do not use any more states for training from the current trajectory
        #    if new_rewards[ind] > -10000:
        #        agent.mem.push(new_states[ind], action, new_rewards[ind+1], new_states[ind+1])
        #    else: 
        #        break

        if np.sum(new_rewards) > -12000:
            for ind in range(len(new_states)-1):
                #If reward is too small then do not use any more states for training from the current trajectory
                if new_rewards[ind] > -10000:
                    agent.mem.push(new_states[ind], action, new_rewards[ind+1], new_states[ind+1])
                else: 
                    break

        agent.learn(batch_size)

        #Information for graphs
        last_reward = new_rewards[len(new_rewards)-1]
        eps_reward = np.sum(new_rewards)

        if last_reward >= -10000:
            rewards.append(last_reward)
            avg_rewards.append(np.mean(rewards[-20:]))

        #Save the best paramters encountered so far
        if last_reward > best_parameters[1]:
            best_parameters[0] = action
            best_parameters[1] = last_reward
            best_parameters[2] = step*(episode + 1)
            exit = True


plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
#plt.plot(normalized)
#plt.show()
#print(best_parameters)
#env.reset(20)
#env.step(best_parameters[0])
#env.render()
#print(all_step)
#plt.plot(all_step)
#plt.show()

#plt.plot(inlook)
#plt.show()
