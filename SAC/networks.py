import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                   name, chkpt_dir='tmp/_sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        # self.q = nn.Linear(self.fc2_dims, 1)

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 2),
                               nn.MaxPool2d(2, 2),
                               nn.Conv2d(32, 64, 2),
                               nn.MaxPool2d(2, 2),

                               ResBlock(n_filters=64, kernel_size=2),
                               nn.MaxPool2d(4, 4),
                               ResBlock(n_filters=64, kernel_size=2),
                               nn.MaxPool2d(2, 2),
                               ResBlock(n_filters=64, kernel_size=2),
                               nn.MaxPool2d(2, 2),
                               ResBlock(n_filters=64, kernel_size=2),
                               nn.MaxPool2d(2, 2),

                               nn.Conv2d(64, 128, 2),
                               nn.Flatten(),

                               nn.Linear(512, 256),
                               nn.ReLU()
                               )

        self.fc = nn.Linear(256+1,64)
        self.q = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # action_value = self.fc1(T.cat([state, action], dim=1))
        # action_value = F.relu(action_value)
        # action_value = self.fc2(action_value)
        # action_value = F.relu(action_value)

        # q1 = self.q(action_value)
        conv = self.conv(state)
        q1 = self.fc(T.cat([conv, action], dim=1))
        q1 = self.q(q1)

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_actions,
                   n_actions, name, chkpt_dir='tmp/_sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_actions = max_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.actor = nn.Sequential(nn.Conv2d(3, 32, 2),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 64, 2),
                                   nn.MaxPool2d(2, 2),

                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(4, 4),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),

                                   nn.Conv2d(64, 128, 2),
                                   nn.Flatten(),

                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128)
                                   )
        self.mu = nn.Linear(128, self.n_actions)
        self.sigma = nn.Linear(128, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # prob = self.fc1(state)
        # prob = F.relu(prob)
        # prob = self.fc2(prob)
        # prob = F.relu(prob)
        prob = self.actor(state)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu,sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1,keepdim=True)

        return action, log_probs   # mu, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 name, chkpt_dir='tmp/_sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        # self.v = nn.Linear(self.fc2_dims, 1)

        self.value = nn.Sequential(nn.Conv2d(3, 32, 2),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 64, 2),
                                   nn.MaxPool2d(2, 2),

                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(4, 4),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),
                                   ResBlock(n_filters=64, kernel_size=2),
                                   nn.MaxPool2d(2, 2),

                                   nn.Conv2d(64, 128, 2),
                                   nn.Flatten(),

                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1)
                                   )


        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # state_value = self.fc1(state)
        # state_value = F.relu(state_value)
        # state_value = self.fc2(state_value)
        # state_value = F.relu(state_value)
        #
        # v = self.v(state_value)
        v = self.value(state)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ResBlock(nn.Module):
    def __init__(self, n_filters, kernel_size):
        """
        Инициализация кастомного резнетовского блока
        :param n_filters: (int) количество фильтров сверточного слоя
        :param kernel_size: (int) размер ядра свертки
        """
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.b1 = nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding='same')

        self.b2 = nn.BatchNorm2d(self.n_filters, eps=0.001, momentum=0.99)
        self.b3 = nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding='same')
        self.b4 = nn.BatchNorm2d(self.n_filters, eps=0.001, momentum=0.99)

    def forward(self, x):
        '''
        Forward propagation
        :param x: input
        :return: output
        '''
        residual = x
        y = F.relu(self.b1(x))
        y = self.b2(y)
        y = F.relu(self.b3(y))
        y = self.b4(y)
        y += residual
        y = F.relu(y)
        return y









