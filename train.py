from env.pid_env import PidEnvSingle
import torch
import numpy as np
from ddpg import Agent
from OUNoise import Noise
import matplotlib.pyplot as plt

batch_size = 128
rewards = []
avg_rewards = []
env = PidEnvSingle()
agent = Agent(num_states=2, num_actions=3, gamma=0)
noise = Noise(num_actions=3)
zeros = [0]
normalized = []
all_steps = [-1]*10
inlook = []
metalearn = False

setpoints = []

if metalearn == True:
    for i in range(10):
        setpoints.append(np.random.random()*100)
    agent.metalearn(setpoints)


for episode in range(1):
    noise.reset()
    eps_reward = 0
    setpoint = np.random.random()*500
    exit = False
    print(episode)
    best_parameters = [(0,0,0), 0, 0]
    step = 0
    done = False
    for i in range(800):
        print(step)
        step += 1
        setpoint = 20
        state = env.reset(setpoint)
        action_before = agent.get_action(state)
        action = noise.get_action(action_before, step)
        if np.isnan(action[0]).any():
            print(action)
            print(action_before)
            print(state)
            for i in agent.actor.parameters():
                print(i)
            exit = True
            break

        new_state, reward = env.step(action)
        if reward > -1000000:
            agent.mem.push(state, action, reward, new_state)
        if reward > -100:
            normalized.append(reward)
        if reward > best_parameters[1]:
            best_parameters[0] = action
            best_parameters[1] = reward
            best_parameters[2] = step*(episode + 1)
            all_steps[episode] = (step*episode+step)
            exit = True
        rewards.append(reward)
        if reward > -10000:
            inlook.append(reward)
            avg_rewards.append(np.mean(inlook[-10:]))
        if reward > 0:
            done = True



            

#        if reward > -5:
#            env.render()
        agent.learn(batch_size)

        state = new_state

#plt.plot(rewards)
#plt.plot(avg_rewards)
#plt.plot(zeros)
#plt.xlabel('Episode')
#plt.ylabel('Reward')
#plt.show()
plt.plot(avg_rewards)
plt.show()
print(best_parameters)
env.reset(20)
env.step(best_parameters[0])
env.render()
#print(all_steps)
#plt.plot(all_steps)
#plt.show()
#plt.plot(inlook)
#plt.show()
#plt.plot(avg_rewards)
#plt.show()
