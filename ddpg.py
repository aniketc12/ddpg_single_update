import torch
import torch.nn as nn
import torch.optim as optim
from nn_models import Critic, Actor
from ReplayMemory import Memory
import torch.autograd
from torch.autograd import Variable
import numpy as np


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
    
#    def metalearn(self, setpoints):
#        actor_theta = Actor(self.num_states, self.num_actions)
#        critic_theta = Critic(self.num_states + self.num_actions, self.num_actions)
#        critic_theta_loss = nn.MSELoss()
#        critic_theta_optimizer = optim.Adam(critic_theta.parameters())
#        torch.autograd.detect_anomoly(True)
#        for i in range(10):
#            outer_critic_loss = 0
#            outer_action_loss = 0
#            for setpoint in setpoints:
#                train_mem = Memory(1000)
#                env = PidEnvSingle(setpoint=setpoint)
#                state = env.reset(setpoint)
#                for targetParam, param in zip(critic_theta.parameters(), self.critic.parameters()):
#                    targetParam.data.copy_(param.data)
#
#                for targetParam, param in zip(actor_theta.parameters(), self.actor.parameters()):
#                    targetParam.data.copy_(param.data)
#
#                state_push = state
#                state = np.array(state)
#                state = torch.from_numpy(state).float().unsqueeze(0)
#                action = self.actor.forward(state)
#                action = action.data[0].tolist()
#
#                next_state, reward = env.step(action)
#                inner_loss = -critic_theta(state, actor_theta.forward(state)).mean()
#
#                state = torch.FloatTensor([state_push])
#                action = torch.FloatTensor([action])
#                reward = torch.FloatTensor([reward])
#                next_state = torch.FloatTensor([next_state])
#
#                qval = critic_theta(state, action)
#                next_action = actor_theta(next_state)
#                next_q = critic_theta(next_state, next_action.detach())
#                target_q = reward + self.gamma * next_q
#
#                critic_loss = critic_theta_loss(qval, target_q)
#                critic_theta_optimizer.zero_grad()
#                critic_loss.backward()
#                critic_theta_optimizer.step()
#
#                actor_grads = torch.autograd.grad(inner_loss, actor_theta.parameters(), create_graph=True)
#                for parameter, grad in zip(actor_theta.parameters(), actor_grads):
#                    parameter.data.copy_(parameter - 0.005*grad)
#
#                state = env.reset(setpoint)
#                state_push = state
#                state = np.array(state)
#                state = torch.from_numpy(state).float().unsqueeze(0)
#                action = self.actor.forward(state)
#                action = action.data[0].tolist()
#
#                new_state, reward = env.step(action)
#                outer_action_loss -= critic_theta(state, actor_theta.forward(state)).mean()
#
#                state = torch.FloatTensor(state_push)
#                action = torch.FloatTensor(action)
#                reward = torch.FloatTensor(reward)
#                next_state = torch.FloatTensor(next_state)
#
#                qval = critic_theta(state, action)
#                next_action = actor_theta(next_state)
#                next_q = critic_theta(next_state, next_action.detach())
#                target_q = reward + self.gamma * next_q
#
#                outer_critic_loss -= critic_theta_loss(qvals, target_q)
#
#            self.actorOptimizer.zero_grad()
#            outer_actor_loss.backward()
#            self.actorOptimizer.step()
#
#            self.criticOptimizer.zero_grad()
#            outer_critic_loss.backward()
#            self.criticOptimizer.step()

