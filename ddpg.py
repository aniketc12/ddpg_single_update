import torch
import torch.nn as nn
import torch.optim as optim
from nn_models import Critic, Actor
from ReplayMemory import Memory
import torch.autograd
from torch.autograd import Variable
import numpy as np
from env.pid_env import PidEnvSingle


class Agent:
    def __init__(self, num_states, num_actions, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, mem_size=50000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        self.mem = Memory(mem_size)

        self.actor = Actor(self.num_states, self.num_actions)
        self.actor_target = Actor(self.num_states, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, self.num_actions)
        self.critic_loss = nn.MSELoss()

        for target_parameters, parameters in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(parameters.data)

        for target_parameters, parameters in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(parameters.data)
            

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

    def get_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.data[0].tolist()
        return action

    def learn(self, batch_size):
        if self.mem.length() < batch_size:
            return
        
        states, actions, rewards, next_states = self.mem.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        qnext = self.critic_target.forward(next_states, next_actions.detach())
        q_target = rewards + self.gamma * qnext

        critic_loss = self.critic_loss(q_vals, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for target_parameters, parameters in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_parameters.data.copy_(parameters.data * self.tau + target_parameters.data * (1.0 - self.tau))

        for target_parameters, parameters in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_parameters.data.copy_(parameters.data * self.tau + target_parameters.data * (1.0 - self.tau))
    
    def metalearn(self, train_setpoints):
        actorLearningRate=1e-4
        criticLearningRate=1e-3
        actor_theta = Actor(self.num_states, self.num_actions)
        critic_theta = Critic(self.num_states + self.num_actions, self.num_actions)
        critic_theta_loss = nn.MSELoss()
        actor_theta_optimizer= optim.Adam(self.actor.parameters(), lr=actorLearningRate)
        critic_theta_optimizer = optim.Adam(self.critic.parameters(), lr=criticLearningRate)
        
        for outer in range(100):
            print(outer)
            outer_critic_loss = 0
            outer_actor_loss = 0
            for i in range(len(train_setpoints)):
                train_mem = Memory(10)
                env = PidEnvSingle(train_setpoints[i])
                state = env.reset(train_setpoints[i])
                eps_reward = 0
                inner_loss = 0
                #Before every trajectory, reinitialze theta parameters to original parameters
                for targetParam, param in zip(critic_theta.parameters(), self.critic.parameters()):
                    targetParam.data.copy_(param.data)

                for targetParam, param in zip(actor_theta.parameters(), self.actor.parameters()):
                    targetParam.data.copy_(param.data)

                for step in range(10):
                    state_push = state
                    state = np.array(state)
                    state = torch.from_numpy(state).float().unsqueeze(0)
                    action = self.actor.forward(state)
                    action = action.data[0].tolist()
                    new_state, reward = env.step(action)
                    train_mem.push(state_push, action, reward, new_state)

                    #Inner loss is negative sum of expected reward which is known using the critic theta network
                    inner_loss -= critic_theta(state, actor_theta(state)).mean()

                    state = env.reset(train_setpoints[i])


                #update critic theta's parameters using mean squared error loss
                size = int(step/2) if step > 10 else step
                states, actions, rewards, next_states = train_mem.sample(size)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                qvals = critic_theta(states, actions)
                next_actions = actor_theta(next_states)
                next_q = critic_theta(next_states, next_actions.detach())
                target_q = rewards + self.gamma * next_q
                critic_loss = critic_theta_loss(qvals, target_q)
                critic_theta_optimizer.zero_grad()
                critic_loss.backward()
                critic_theta_optimizer.step()

                #Update actor theta using inner loss(which is the negative sum of expected reward) as the loss function
                actor_theta_optimizer.zero_grad()
                actor_grads = torch.autograd.grad(inner_loss, actor_theta.parameters(), create_graph=True)
                for parameter, grad in zip(actor_theta.parameters(), actor_grads):
                    parameter.data.copy_(parameter - 0.005*grad)

                train_mem = Memory(10)
                env = PidEnvSingle(train_setpoints[i])
                state = env.reset(train_setpoints[i])
                eps_reward = 0
                done = False
                for step in range(10):
                    state_push = state
                    state = np.array(state)
                    state = torch.FloatTensor(state).unsqueeze(0)
                    action = actor_theta(state).data[0].tolist()
                    new_state, reward = env.step(action)
                    train_mem.push(state_push, action, reward, new_state)

                    #Inner loss is negative sum of expected reward which is known using the critic theta network
                    outer_actor_loss -= critic_theta(state, actor_theta(state)).mean()

                    state = env.reset(train_setpoints[i])

                #update critic theta's parameters using mean squared error loss
                size = int(step/2) if step > 10 else step
                states, actions, rewards, next_states = train_mem.sample(size)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                qvals = critic_theta(states, actions)
                next_actions = actor_theta(next_states)
                next_q = critic_theta(next_states, next_actions.detach())
                target_q = rewards + self.gamma * next_q
                outer_critic_loss += critic_theta_loss(qvals, target_q)

            self.actor_optimizer.zero_grad()
            outer_actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            outer_critic_loss.backward()
            self.critic_optimizer.step()

