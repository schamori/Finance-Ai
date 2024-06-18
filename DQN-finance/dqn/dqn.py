import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections

"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim):
        super(QNetworkCNN, self).__init__()

        self.conv_1 = nn.Conv2d(1, 8, kernel_size=8, stride=3)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=2, stride=1)

        self.fc_input_size = self._calculate_fc_input((1, 84, 84))

        self.fc_1 = nn.Linear(self.fc_input_size, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def _calculate_fc_input(self, input_shape):
        # Perform a forward pass with a dummy input to calculate the flattened size
        dummy_input = torch.zeros(1, *input_shape)
        x = self.conv_1(dummy_input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x.numel()

    def forward(self, inp):
        x = F.relu(self.conv_1(inp))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        self.state.extend(state)
        self.action.extend(action)
        self.rewards.extend(reward)
        self.is_done.extend([False for _ in range(len(state )- 1)])
        self.is_done.append(True)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.state)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(np.array(self.state))[idx].to(device), torch.LongTensor(np.array(self.action))[idx].to(device), \
               torch.Tensor(np.array(self.state))[1+np.array(idx)].to(device), torch.Tensor(np.array(self.rewards))[idx].to(device), \
               torch.Tensor(np.array(self.is_done))[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_actions(model, env, states, eps):
    states = torch.Tensor(np.array(states)).to(device)
    with torch.no_grad():
        values = model(states)

    # select a random action wih probability eps

    actions = [
        np.random.randint(0, env.action_space.n) if random.random() <= eps else np.argmax(value)
        for value in values.cpu().numpy()
    ]

    return actions


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        states = env.reset()
        states = torch.Tensor(np.array(states)).to(device)
        with torch.no_grad():
            values = Qmodel(states)
        actions = [np.argmax(value) for value in values.cpu().numpy()]
        reward = env.calculate_rewards(actions)
        perform += np.array(reward).mean()
    Qmodel.train()
    action_map = [-1, 0, 1]
    actions = [action_map[action] for action in actions]
    return perform/repeats, actions



def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())



