import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from memory import PPOMemory


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

class ActorNetwork(nn.Module):
    def __init__(self,n_actions, input_dims, alpha,
                 chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')

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
                                   nn.Linear(256, n_actions),
                                   nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,  chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')

        self.critic = nn.Sequential(nn.Conv2d(3, 32, 2),
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
                                   nn.Linear(256, 1),
                                   nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions = n_actions,
                                  input_dims = input_dims,
                                  alpha = alpha)
        self.critic = CriticNetwork(input_dims, alpha = alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
            reward_arr, done_arrs, batches = self.memory.generate_butches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype = np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]* \
                                    (1-int(done_arrs[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weight_probs = advantage[batch] * prob_ratio
                weight_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weight_probs, weight_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()