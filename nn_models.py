import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_size)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.output(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_size)
    
#    def forward(self, state):
#        x = torch.tanh(self.linear1(state))
#        x = torch.tanh(self.linear2(x))
#        x = (self.output(x))
#        return x
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        x = (self.output(x))
        return x
